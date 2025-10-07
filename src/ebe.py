import numpy as np


import gc
import json
import time

import numpy as np
import pandas as pd
import torch

# from baseline_models import get_models_and_baseline_metric
from forecaster import forecast_generation
from candidates import Candidate
from evolution import breed_and_mutate
from forecaster import forecast_generation
from instance_sampling import sample_data
from scoring import (
    check_higher_than_baseline,
    compute_and_log_p_above_goal,
    convex_lb_discard,
    score_individuals,
)
from utils import init_global_seed, make_repro_context
import data_preprocessing as dp


import pandas as pd
from architecture_generator import create_model_from_row
# region Generations

class Population:
    """Maintain a population of neural architectures under the EBE loop."""

    def __init__(
        self,
        search_space,
        size,
        starting_instances=0.1,
        seed= None,
        task_type: str = "classification",
    ) -> None:
        self.seed = int(seed if seed is not None else np.random.default_rng().integers(0, 1_000_000))
        init_global_seed(self.seed)
        self.repro = make_repro_context(self.seed)

        self.task_type = task_type
        self.search_space = search_space
        self.max_individuals = size
        self.size = size
        self.starting_instances = starting_instances
        self.individuals_created = 0
        self.generations_completed = 0
        self.candidates = {}
        self.spawn_new_candidates(self.search_space)  # use the same spawn logic
        self.initial_ledger = self.build_ledger().copy(deep=True)
        self.generation_logs = []

        
    def spawn_candidates(self):
        candidates_pool = {}
        rng = self.repro.np_rng
        for i in range(self.size):
            arch = self.search_space.sample_architecture(rng=rng)
            model = self.search_space.create_model(arch, task_type=self.task_type)
            candidates_pool[i] = Candidate(
                model,
                arch,
                starting_instances=self._initial_budget(),
                identifier=i,
            )
            self.individuals_created += 1
        return candidates_pool

    

    def train_generation(self, X_train, y_train, 
                    training_mode='oe', 
                    X_val=None, y_val=None, time_budget=None,
                    **kwargs):
        """Train all candidates."""


        def train_one_candidate(candidate, train_loader, val_loader,
                        num_epochs=1, val_frequency=1, measure_wall_time: bool = True):
            """
            Train a single candidate, logging metrics per batch directly into Candidate.metrics.
            Validation is mandatory and runs every `val_frequency` batches.
            Returns total epoch time (sum of batch times).
            """
            model = candidate.model
            model.train()
            epoch_time_acc = 0.0
            total_batches = 0

            for epoch in range(num_epochs):
                n_batches_epoch = 0
                for features, labels in train_loader: # one batch
                    start = time.time() if measure_wall_time else None

                    # forward + backward
                    features, labels = features.to(model.device), labels.to(model.device)
                    if features.dim() > 2:
                        features = features.view(features.size(0), -1)

                    model.optimizer.zero_grad()
                    outputs = model(features)
                    loss = model.criterion(outputs, labels)
                    loss.backward()
                    model.optimizer.step()

                    batch_time = time.time() - start
                    epoch_time_acc += batch_time
                    total_batches += 1

                    with torch.no_grad():
                        _, predicted = torch.max(outputs, 1)
                        train_acc = (predicted == labels).float().mean().item()

                    # log train metrics
                    candidate.log_metric("train", "loss", loss.item())
                    candidate.log_metric("train", "acc", train_acc)
                    batch_wall = (time.time() - start) if measure_wall_time else None
                    candidate.log_effort(batch_wall_time=batch_wall)
                    
                    n_batches_epoch += 1

                    # validation (always required)
                    if total_batches % val_frequency == 0:
                        model.eval()
                        correct, total, val_loss_sum = 0, 0, 0.0
                        with torch.no_grad():
                            for v_features, v_labels in val_loader:
                                v_features, v_labels = v_features.to(model.device), v_labels.to(model.device)
                                if v_features.dim() > 2:
                                    v_features = v_features.view(v_features.size(0), -1)

                                v_outputs = model(v_features)
                                v_loss = model.criterion(v_outputs, v_labels)
                                val_loss_sum += v_loss.item() * v_features.size(0)

                                _, v_pred = torch.max(v_outputs, 1)
                                correct += (v_pred == v_labels).sum().item()
                                total += v_labels.size(0)

                        val_acc = correct / total
                        val_loss = val_loss_sum / total
                        candidate.log_metric("val", "loss", val_loss)
                        candidate.log_metric("val", "acc", val_acc)
                        model.train()
                        

            # store epoch batches
            candidate.metrics.setdefault("epoch_batches", [])
            candidate.metrics["epoch_batches"].append(n_batches_epoch)

            return n_batches_epoch

        """Train all candidates in the population."""
        start_time = time.time()
        for i in list(self.candidates.keys()):
            # Check if time budget exceeded
            if time_budget is not None and time.time() - start_time >= time_budget:
                print(f"[Train] Time budget of {time_budget}s exceeded, stopping training.")
                break

            candidate = self.candidates[i]
            model = candidate.model
            batch_size = candidate.batch_size

            n_instances = min(candidate.n_instances[-1], len(X_train))

            # Use the shared RNG to choose indices
            X_sampled, y_sampled = sample_data(
                X_train, y_train, n_instances,
                mode="absolute",
                seed=None,                   
                task_type=self.task_type,
                rng=self.repro.np_rng         # NEW
            )

            train_loader = dp.create_dataloader(
                X=X_sampled, y=y_sampled,
                batch_size=batch_size,
                generator=self.repro.torch_gen,
                seed_worker=self.repro.seed_worker
            )

            if training_mode == 'oe':
                assert X_val is not None and y_val is not None, \
                    "Validation data must be provided for one-epoch training."

                val_loader = dp.create_dataloader(
                        X=X_val, y=y_val,
                        batch_size=batch_size,
                        generator=self.repro.torch_gen,
                        seed_worker=self.repro.seed_worker
                    )

                epoch_batches = train_one_candidate(
                    candidate, train_loader, val_loader,
                    num_epochs=1, val_frequency=1
                )
                candidate.metrics.setdefault("epoch_batches", [])
                candidate.metrics["epoch_batches"].append(epoch_batches)

                # update budget + epoch counter
                candidate.next_anchor(growth=1.4, max_cap=len(X_train))
                candidate.epochs_trained += 1

    def spawn_new_candidates(self, search_space):
        self.size = len(self.candidates)
        n_new = self.max_individuals - self.size
        if n_new <= 0:
            return

        rng = self.repro.np_rng
        new_generation = {}

        if self.size == 0:
            # First generation: fill entirely with fresh randoms
            for _ in range(n_new):
                arch = search_space.sample_architecture(rng=rng)
                model = search_space.create_model(arch, task_type=self.task_type)
                new_generation[self.individuals_created] = Candidate(
                    model, arch,
                    starting_instances=self.starting_instances,
                    id_counter=self.individuals_created + 1
                )
                self.individuals_created += 1
        else:
            # Mixed: some randoms, some bred
            n_basic = n_new // 2
            n_adv = n_new - n_basic

            for _ in range(n_basic):
                arch = search_space.sample_architecture(rng=rng)
                model = search_space.create_model(arch, task_type=self.task_type)
                new_generation[self.individuals_created] = Candidate(
                    model, arch,
                    starting_instances=self.starting_instances,
                    id_counter=self.individuals_created + 1
                )
                self.individuals_created += 1

            for _ in range(n_adv):
                child_arch = breed_and_mutate(self.candidates, rng=rng)
                child_model = search_space.create_model(child_arch, task_type=self.task_type)
                new_generation[self.individuals_created] = Candidate(
                    child_model, child_arch,
                    id_counter=self.individuals_created + 1,
                    starting_instances=self.starting_instances
                )
                self.individuals_created += 1

        self.candidates.update(new_generation)
        self.size = len(self.candidates)




    def prune_candidates(self,
                    baseline_metric,
                    base_drop=0.1,
                    max_drop=0.3,
                    elite_fraction=0.2,
                    incubator_fraction=0.1,
                    min_survivors=5,
                    min_batches_protected=5):
        """
        Softer unified pruning with per-candidate protection.
        - Any candidate with < min_batches_protected is fully shielded.
        - Convex LB + probability drop only apply to candidates with enough training.
        - Larger elite and incubator buffers give more safety.
        """

        n = len(self.candidates)
        if n <= min_survivors:
            self.worst_individuals = []
            print(f"[Prune] total={n}, below survivor floor, no pruning.")
            return

        b_ref = max(c.n_instances[-1] for c in self.candidates.values())

        hopeless, survivors, protected = [], [], []
        for k, cand in self.candidates.items():
            if cand.batches_trained < min_batches_protected:
                # hard shield: too few batches, cannot be dropped
                protected.append(k)
                continue

            # --- NEW CI PROTECTION ---
            ci_high = cand.metrics.get("forecast_CI_high", cand.metrics.get("forecasted_val_acc", 0.0))
            if ci_high >= baseline_metric:
                # even if forecast mean is weak, upper CI says it *might* win → protect
                protected.append(k)
                continue

            # --- hopeless check (convex LB discard) ---
            if  convex_lb_discard(cand, baseline_metric, b_ref):
                hopeless.append(k)
            else:
                survivors.append(k)


        # If no candidate has enough anchors, skip pruning entirely
        if not survivors:
            self.worst_individuals = []
            print(f"[Prune] Skipped: all candidates protected or under-trained.")
            return

        # Drop fraction
        frac = min(max_drop, base_drop + 0.01 * self.generations_completed)
        n_drop = int(len(survivors) * frac)

        # Rank by adjusted probability (boost by training effort)
        ranked_by_prob = sorted(
            [
                (
                    k,
                    self.candidates[k].metrics.get("p_above_goal", 0.0)
                    * (1.0 + 0.2 * self.candidates[k].batches_trained)
                )
                for k in survivors
            ],
            key=lambda x: x[1],
            reverse=True
        )

        # Elite buffer
        elite_count = max(1, int(elite_fraction * n))
        elites = {
            k for k, c in sorted(
                self.candidates.items(),
                key=lambda kv: kv[1].get_metric("val", "acc", last_only=True) or 0.0,
                reverse=True
            )[:elite_count]
        }

        # Incubator buffer
        incubator_count = max(1, int(incubator_fraction * n))
        incubators = {k for k, _ in ranked_by_prob[-incubator_count:]}

        # Survivors by prob
        n_keep = max(min_survivors, len(survivors) - n_drop)
        keep_prob = {k for k, _ in ranked_by_prob[:n_keep]}

        # Combine everything
        keep = keep_prob | elites | incubators | set(protected)
        worst = [k for k in self.candidates if k not in keep] + hopeless
        worst = list(set(worst))

        self.worst_individuals = worst

        print(f"[Prune] total={n}, protected={len(protected)}, hopeless={len(hopeless)}, "
            f"dropped={len(worst)}, kept={n - len(worst)}")


    def drop_worst_individuals(self):
        # Move all worst models to CPU first
        for idx in self.worst_individuals:
            candidate = self.candidates[idx]
            if hasattr(candidate.model, "cpu"):
                candidate.model = candidate.model.cpu()
        
        # Remove worst individuals
        for idx in self.worst_individuals:
            del self.candidates[idx]
        
        self.size = len(self.candidates)  # Update the count

        # Clean up GPU memory once
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def run_generation(self,
                    X_train, y_train, X_val, y_val,
                    baseline_metric,
                    base_drop=0.2,
                    max_drop=0.5,
                    track_all_models=False,
                    time_budget=None):
        """
        Run a single evolutionary generation:
        - Spawns new candidates (random + bred)
        - Trains each candidate (OE)
        - Forecasts validation accuracy
        - Scores and prunes
        - Logs rich generation-level telemetry
        """

        import numpy as np
        log = {"gen": self.generations_completed + 1}
        gen_start = time.time()

        # === PHASE 0: Spawn ===
        before_spawn = len(self.candidates)
        self.spawn_new_candidates(self.search_space)
        after_spawn = len(self.candidates)
        log["spawned"] = after_spawn - before_spawn
        log["population_before_train"] = before_spawn
        log["population_after_spawn"] = after_spawn

        # === PHASE 1: Train ===
        before_train = len(self.candidates)
        self.train_generation(X_train, y_train,
                            training_mode="oe",
                            X_val=X_val, y_val=y_val,
                            time_budget=time_budget)
        log["trained"] = before_train

        # --- compute total training time in this generation ---
        total_epoch_times = []
        for cand in self.candidates.values():
            if "epoch_time" in cand.metrics:
                total_epoch_times.extend(cand.metrics["epoch_time"])
        compute_time = float(sum(total_epoch_times)) if total_epoch_times else 0.0
        log["compute_this_gen"] = compute_time

        # === PHASE 2: Forecast & Scoring ===
        forecast_generation(self.candidates,
                            dataset_size=len(X_train),
                            min_val_points=5,
                            extra_full_passes=5)
        check_higher_than_baseline(self.candidates, baseline_metric)

        dynamic_goal = max(
            baseline_metric or 0.0,
            max((c.get_metric("val", "acc", last_only=True) or 0.0)
                for c in self.candidates.values()),
            max((c.metrics.get("forecasted_val_acc", 0.0) or 0.0)
                for c in self.candidates.values())
        )

        compute_and_log_p_above_goal(self.candidates, goal_metric=dynamic_goal)
        score_individuals(self.candidates)

        # === PHASE 3: Pruning ===
        survivors_before = len(self.candidates)
        self.prune_candidates(
            baseline_metric=dynamic_goal,
            base_drop=base_drop,
            max_drop=max_drop,
            elite_fraction=0.1,
            min_survivors=5
        )
        self.drop_worst_individuals()
        survivors_after = len(self.candidates)

        log["hybrid_dropped"] = survivors_before - survivors_after
        log["survivors"] = survivors_after

        # === PHASE 4: Extended Statistics ===
        # Gather population-level telemetry
        val_accs = [c.get_metric("val", "acc", last_only=True) or 0.0 for c in self.candidates.values()]
        fcsts = [c.metrics.get("forecasted_val_acc", 0.0) for c in self.candidates.values()]
        p_above = [c.metrics.get("p_above_goal", 0.0) for c in self.candidates.values()]
        ci_highs = [c.metrics.get("forecast_CI_high", 0.0) for c in self.candidates.values()]
        ci_lows = [c.metrics.get("forecast_CI_low", 0.0) for c in self.candidates.values()]
        efforts = [sum(c.efforts) for c in self.candidates.values() if c.efforts]

        def _safe_stat(arr, fn):
            if arr is None:
                return None
            # works for both lists and numpy arrays
            if isinstance(arr, (list, tuple)):
                if len(arr) == 0:
                    return None
            elif hasattr(arr, "size"):
                if arr.size == 0:
                    return None
            try:
                return float(fn(arr))
            except Exception:
                return None


        log.update({
            # performance stats
            "val_acc_mean": _safe_stat(val_accs, np.mean),
            "val_acc_std": _safe_stat(val_accs, np.std),
            "val_acc_max": _safe_stat(val_accs, np.max),
            "val_acc_min": _safe_stat(val_accs, np.min),

            # forecast stats
            "fcst_mean": _safe_stat(fcsts, np.mean),
            "fcst_std": _safe_stat(fcsts, np.std),
            "fcst_max": _safe_stat(fcsts, np.max),
            "fcst_min": _safe_stat(fcsts, np.min),

            # probability stats
            "p_above_mean": _safe_stat(p_above, np.mean),
            "p_above_std": _safe_stat(p_above, np.std),
            "p_above_max": _safe_stat(p_above, np.max),
            "p_above_min": _safe_stat(p_above, np.min),

            # CI width
            "ci_high_mean": _safe_stat(ci_highs, np.mean),
            "ci_low_mean": _safe_stat(ci_lows, np.mean),
            "ci_width_mean": _safe_stat(np.array(ci_highs) - np.array(ci_lows), np.mean)
                            if ci_highs and ci_lows else None,

            # effort summary
            "effort_total": _safe_stat(efforts, np.sum),
            "effort_mean": _safe_stat(efforts, np.mean),
            "effort_std": _safe_stat(efforts, np.std),

            # architectural diversity
            "diversity_hidden_layers": _safe_stat(
                [len(c.architecture.get("hidden_layers", [])) for c in self.candidates.values()],
                np.std),
            "diversity_lr": _safe_stat(
                [c.architecture.get("learning_rate", 0.0) for c in self.candidates.values()],
                np.std),
        })

        # --- Decision context (global EU/p/benefit/cost if available) ---
        if hasattr(self, "eu"):
            log["expected_utility_global"] = float(self.eu)
        if hasattr(self, "p"):
            log["p_above_goal_global"] = float(self.p)
        if hasattr(self, "benefit"):
            log["benefit_global"] = float(self.benefit)
        if hasattr(self, "cost"):
            log["cost_global"] = float(self.cost)

        # === PHASE 5: Ledger & cumulative compute ===
        self.size = len(self.candidates)
        self.current_snapshot = self.build_ledger()
        if track_all_models:
            self.cumulative_ledger = (
                pd.concat([self.cumulative_ledger, self.current_snapshot])
                .drop_duplicates(subset="id", keep="last")
            )

        prev_total = self.generation_logs[-1]["cumulative_compute"] if self.generation_logs else 0.0
        log["cumulative_compute"] = prev_total + compute_time
        log["elapsed_gen_time"] = float(time.time() - gen_start)

        # === Store & Return ===
        self.generation_logs.append(log)
        return self.candidates

    def run_ebe(self,
            X_train, y_train, X_val, y_val,
            baseline_metric,
            max_generations=20,
            time_budget=60,
            base_drop=0.1,
            max_drop=0.5,
            track_all_models=False):
        """
        Full EBE loop:
        - Run generations with convex LB + hybrid pruning.
        - After each gen, check worth_training_neural_bayes.
        - If decision flips True, stop EBE and launch extended training.
        - Extended training continues candidate training until:
            (a) baseline surpassed,
            (b) time budget exhausted,
            (c) convex LB says hopeless.
        """

        self.current_snapshot = self.initial_ledger
        self.cumulative_ledger = self.initial_ledger if track_all_models else None

        start_time = time.time()
        triggered = False
        best_cand = None

        # === Main EBE loop ===
        for gen in range(max_generations):
            elapsed = time.time() - start_time
            if elapsed >= time_budget:
                print(f"[EBE] Time budget exceeded at gen {gen+1}, elapsed={elapsed:.2f}s")
                break

            print(f"\n=== Generation {gen+1} ===")

            self.candidates = self.run_generation(
                X_train, y_train, X_val, y_val,
                baseline_metric=baseline_metric,
                base_drop=base_drop,
                max_drop=max_drop,
                track_all_models=track_all_models, 
                time_budget=time_budget - elapsed
            )
            self.generations_completed += 1

            # decision check after each generation
            self.decision, self.eu, self.p, self.benefit, self.cost = \
                self.worth_training_neural_bayes(baseline_metric=baseline_metric)

            if self.decision:
                print(f"[EBE] At gen {gen+1}, decision flipped: Worth training NN")
                best_cand = max(self.candidates.values(),
                                key=lambda c: c.metrics.get("forecasted_val_acc", 0.0))
                triggered = True
                break

        # print("EBE process completed.")

        # === Post-EBE extended training if triggered ===
        self.ebe_loop_time = time.time() - start_time  # total EBE search time

        if triggered and best_cand is not None:
            self.extend_selected_candidate(
                best_cand=best_cand,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                baseline_metric=baseline_metric,
                time_budget=time_budget,
                start_time=start_time
            )
        elif not triggered:
            print("[EBE] No evidence found that NN is worth training. Skipping extension.")

        return self.current_snapshot


    # def fidelity_training(self, X_train, y_train, X_val, y_val, 
    #                       X_test=None, y_test=None):
    #     # === Fidelity check ===
    #     print("Training top by ES (fidelity check)")
    #     self.fidelity_ledger = self.fidelity_from_ledger(
    #         ledger_df=self.current_snapshot,
    #         X_train=X_train, y_train=y_train,
    #         X_val=X_val, y_val=y_val,
    #         top_fraction=1, # all of them,
    #     )
    #     return self.fidelity_ledger

    

    def build_ledger(self, export_as='pandas'):
        def _to_scalar(val):
            if isinstance(val, list):
                return val[-1] if val else np.nan
            return val
        current_candidates = []
        active_individuals = self.candidates.keys()
        for i in active_individuals:
            candidate = self.candidates[i]
            cand_dict = candidate.build_dict()
            current_candidates.append(cand_dict)

        if export_as == 'pandas':
            df = pd.DataFrame(current_candidates)
            df = df.sort_values(by='score', ascending=False)
            return df.copy(deep=True)

        elif export_as == 'json':
            return json.dumps(current_candidates, indent=4)


    def worth_training_neural_bayes(self, baseline_metric, cost_scale=1.0, tol=0.0):
        best_cand = max(self.candidates.values(), key=lambda c: c.metrics.get("p_above_goal", 0.0))

        p    = best_cand.metrics.get("p_above_goal", 0.0)
        fcst = best_cand.metrics.get("forecasted_val_acc", 0.0)
        benefit = max(0.02, fcst - baseline_metric)

        # effort-based horizon
        e_future = best_cand.metrics.get("forecast_horizon_time") or 0.0   # effort units now
        e_spent  = best_cand.cumulative_effort[-1] if getattr(best_cand, "cumulative_effort", None) else 0.0
        projected_remaining_effort = max(0.0, e_future - e_spent)

        # normalized cost proxy in [0,1)
        norm_cost = projected_remaining_effort / (projected_remaining_effort + 1.0)

        gen = getattr(self, "generations_completed", 0)
        exploration_weight = max(0.3, 1.0 - 0.01 * gen)

        EU = (p * benefit * exploration_weight) - (1 - p) * 0.05 - cost_scale * 0.1 * norm_cost
        return EU > tol, EU, p, benefit, projected_remaining_effort



    
    def final_decision(self):
        return {
            'ShouldYouEvenNN?': self.decision,
            'Exp. Utility': self.eu,
            'p_above_goal': self.p,
            'benefit': self.benefit,
            'cost': self.cost
        }
    
    def export_generation_logs(self, path="generation_logs.csv"):
        import pandas as pd
        df = pd.DataFrame(self.generation_logs)
        df.to_csv(path, index=False)
        return df
    

    def fidelity_from_ledger(self, ledger_df,
                         X_train, y_train,
                         X_val, y_val,
                         top_fraction=0.1,
                         max_time=None):
        """
        Rebuild a subset of models from a ledger and retrain them 
        with early stopping to obtain their actual validation performance.

        Used purely for forecast-vs-actual benchmarking.
        """

        # --- Select top candidates by forecasted validation accuracy ---
        ranked = ledger_df.sort_values("forecasted_val_acc", ascending=False)
        n_keep = max(1, int(len(ranked) * top_fraction))
        chosen = ranked.head(n_keep)

        fidelity_records = []

        for idx, row in enumerate(chosen.itertuples(), 1):
            print(f"[Fidelity] Candidate {idx}/{n_keep} (id={row.id})")

            # --- Rebuild model from ledger row ---
            model = create_model_from_row(
                row._asdict(),
                input_size=self.search_space.input_size,
                output_size=self.search_space.output_size,
                task_type=self.task_type,
            )

            # --- Minimal candidate wrapper for logging ---
            cand = Candidate(model, row._asdict(), id_counter=row.id)

            # --- Data loaders ---
            train_loader = dp.create_dataloader(
                X=X_train, y=y_train,
                batch_size=cand.batch_size,
                generator=self.repro.torch_gen,
                seed_worker=self.repro.seed_worker
            )
            val_loader = dp.create_dataloader(
                X=X_val, y=y_val,
                batch_size=cand.batch_size,
                generator=self.repro.torch_gen,
                seed_worker=self.repro.seed_worker
            )

            # --- Train with early stopping ---
            (
                best_train_loss,
                best_train_acc,
                best_val_loss,
                best_val_acc,
                lc
            ) = model.early_stopping_train(
                cand, train_loader, val_loader,
                task_type=self.task_type,
                max_time=max_time,
                return_lc=True
            )

            # --- Store minimal fidelity record ---
            fidelity_records.append({
                "id": cand.id,
                "forecasted_val_acc": getattr(row, "forecasted_val_acc", None),
                "forecasted_CI_high": getattr(row, "forecast_CI_high", None),
                "forecasted_CI_low": getattr(row, "forecast_CI_low", None),
                "forecast_horizon_time": getattr(row, "forecast_horizon_time", None),
                "fidelity_val_acc": best_val_acc,
                "fidelity_val_loss": best_val_loss,
                "fidelity_train_acc": best_train_acc,
                "fidelity_train_loss": best_train_loss,
                "learning_curve": lc,
            })

        fidelity_ledger = pd.DataFrame(fidelity_records)
        fidelity_ledger = fidelity_ledger.dropna(subset=["fidelity_val_acc"]).sort_values(
            "fidelity_val_acc", ascending=False, na_position="last"
        )

        return fidelity_ledger


    def extend_selected_candidate(self, best_cand,
                              X_train, y_train, X_val, y_val,
                              baseline_metric, time_budget, start_time):
        """
        Post-EBE extended training phase.
        Runs one continuous early-stopping session with the remaining global time.
        """

        print("[Post-EBE Extended Training] Starting refinement phase")

        # loaders
        train_loader = dp.create_dataloader(
            X=X_train, y=y_train,
            batch_size=best_cand.batch_size,
            generator=self.repro.torch_gen,
            seed_worker=self.repro.seed_worker
        )
        val_loader = dp.create_dataloader(
            X=X_val, y=y_val,
            batch_size=best_cand.batch_size,
            generator=self.repro.torch_gen,
            seed_worker=self.repro.seed_worker
        )

        model = best_cand.model
        self.extension_log = []
        ext_start = time.time()
        elapsed_global = time.time() - start_time
        remaining_time = max(time_budget - elapsed_global, 10.0)

        print(f"[Extension] Launching ES training (remaining {remaining_time:.1f}s)")

        # run a single ES training session
        best_train_loss, best_train_acc, best_val_loss, best_val_acc, lc = \
            model.early_stopping_train(
                candidate=best_cand,
                train_loader=train_loader,
                val_loader=val_loader,
                task_type=self.task_type,
                patience=30,
                tol=1e-4,
                max_time=remaining_time,
                return_lc=True
            )

        surpassed = best_val_acc is not None and best_val_acc >= baseline_metric

        # log snapshot
        self.extension_log.append({
            "elapsed_global": elapsed_global,
            "remaining_time": remaining_time,
            "val_acc": best_val_acc,
            "val_loss": best_val_loss,
            "train_acc": best_train_acc
        })
        print(f"[Extension] val_acc={best_val_acc:.4f} | "
            f"Baseline={baseline_metric:.4f} | Surpassed? {surpassed}")

        # timing summary
        self.extended_training_time = time.time() - ext_start
        total_elapsed = self.ebe_loop_time + self.extended_training_time

        print(f"[Post-EBE Summary] EBE loop: {self.ebe_loop_time:.2f}s | "
            f"Extension: {self.extended_training_time:.2f}s | "
            f"Total: {total_elapsed:.2f}s")

        # structured result
        self.extension_result = {
            "surpassed": surpassed,
            "final_val_acc": best_val_acc,
            "baseline": baseline_metric,
            "elapsed_total": total_elapsed
        }

        self.extension_summary = {
            "candidate_id": best_cand.id,
            "surpassed_baseline": surpassed,
            "final_val_acc": float(best_val_acc or 0.0),
            "final_val_loss": float(best_val_loss or 0.0),
            "best_train_acc": float(best_train_acc or 0.0),
            "best_train_loss": float(best_train_loss or 0.0),
            "baseline_metric": float(baseline_metric or 0.0),
            "ebe_loop_time": float(getattr(self, "ebe_loop_time", 0.0)),
            "extended_training_time": float(self.extended_training_time),
            "total_elapsed": float(total_elapsed),
            "extension_steps": 1
        }

        # ledger update
        self.current_snapshot = self.build_ledger().copy(deep=True)
        return self.current_snapshot


def compare_forecast_vs_fidelity(fidelity_df):
    """
    Compare forecasted validation accuracy against actual
    early-stopping validation accuracy.

    Returns:
        dict with:
            - MAE   : Mean Absolute Error (|forecast - actual|)
            - Bias  : Mean signed error (actual - forecast)
            - n     : Number of compared candidates
            - details : DataFrame with id, forecast, actual, delta
    """

    import numpy as np
    import pandas as pd

    # --- Basic validation ---
    required_cols = {"id", "forecasted_val_acc", "fidelity_val_acc"}
    missing = required_cols - set(fidelity_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in fidelity_df: {missing}")

    df = fidelity_df.dropna(subset=["forecasted_val_acc", "fidelity_val_acc"]).copy()

    if len(df) == 0:
        raise ValueError("No valid rows to compare (all NaN).")

    fcst = df["forecasted_val_acc"].astype(float)
    actual = df["fidelity_val_acc"].astype(float)

    deltas = actual - fcst
    mae = float(np.mean(np.abs(deltas)))
    bias = float(np.mean(deltas))

    details = pd.DataFrame({
        "id": df["id"],
        "forecasted_val_acc": fcst,
        "fidelity_val_acc": actual,
        "delta": deltas
    }).sort_values("delta", ascending=False, na_position="last")

    report = {
        "MAE": mae,
        "Bias": bias,
        "n": len(df),
        "details": details
    }

    return report
