import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import gc
import data_preprocessing as dp
from instance_sampling import sample_data

import time
import json


# from baseline_models import get_models_and_baseline_metric
from forecaster import forecast_generation, annotate_probabilities
from candidates import Candidate
from evolution import breed_and_mutate
from scoring import score_individuals, convex_lb_discard, compute_and_log_p_above_goal, check_higher_than_baseline
from utils import init_global_seed, make_repro_context

# region Generations

class Population:
    def __init__(self, search_space, size, 
                 starting_instances=0.1, seed=None, task_type='classification'):
        self.seed = seed or np.random.default_rng().integers(0, 1_000_000)
        init_global_seed(self.seed)                 # one-time init
        self.repro = make_repro_context(self.seed)  # shared RNGs

        self.task_type = task_type
        self.search_space = search_space
        self.max_individuals = size
        self.size = size
        self.starting_instances = starting_instances
        self.individuals_created = 0
        self.candidates = self.spawn_candidates()
        self.initial_ledger = self.build_ledger().copy(deep=True)
        self.generations_completed = 0
        self.generation_logs = []


        

    def spawn_candidates(self):
        candidates_pool = {}
        rng = self.repro.np_rng
        for i in range(self.size):
            arch = self.search_space.sample_architecture(rng=rng)
            model = self.search_space.create_model(arch, task_type=self.task_type)
            candidates_pool[i] = Candidate(model, arch, starting_instances=self.starting_instances, id_counter=i)
            self.individuals_created += 1
        return candidates_pool

    

    def train_generation(self, X_train, y_train, 
                     training_mode='oe', 
                     X_val=None, y_val=None, 
                     **kwargs):
        """Train all candidates."""


        def train_one_candidate(candidate, train_loader, val_loader,
                        num_epochs=1, val_frequency=1):
            """
            Train a single candidate, logging metrics per batch directly into Candidate.metrics.
            Validation is mandatory and runs every `val_frequency` batches.
            Returns total epoch time (sum of batch times).
            """
            model = candidate.model
            model.train()
            total_batches = 0
            epoch_time_acc = 0.0

            for epoch in range(num_epochs):
                n_batches_epoch = 0
                for features, labels in train_loader: # one batch
                    start = time.time()

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
                    candidate.log_effort(batch_time)
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
                        

            # store epoch timing as list
            if "epoch_time" not in candidate.metrics:
                candidate.metrics["epoch_time"] = []
            candidate.metrics["epoch_time"].append(epoch_time_acc)

            return epoch_time_acc



        """Train all candidates in the population."""
        for i in list(self.candidates.keys()):
            candidate = self.candidates[i]
            model = candidate.model
            batch_size = candidate.batch_size

            n_instances = min(candidate.n_instances[-1], len(X_train))

            # Use the shared RNG to choose indices (see sample_data change)
            X_sampled, y_sampled = sample_data(
                X_train, y_train, n_instances,
                mode="absolute",
                seed=None,                    # let rng drive it
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

                epoch_time = train_one_candidate(
                    candidate, train_loader, val_loader,
                    num_epochs=1, val_frequency=1
                )


                # log epoch time
                if "epoch_time" not in candidate.metrics:
                    candidate.metrics["epoch_time"] = []
                candidate.metrics["epoch_time"].append(epoch_time)

                # update budget + epoch counter
                candidate.next_anchor(growth=1.4, max_cap=len(X_train))
                candidate.epochs_trained += 1

            elif training_mode == 'es':
                print(f"Training individual {i+1}/{self.size} with Early Stopping...")
                assert X_val is not None and y_val is not None, \
                    "X_val and y_val must be provided for early stopping training."

                val_loader = dp.create_dataloader(
                    X=X_val, 
                    y=y_val, 
                    batch_size=batch_size, 
                    generator=g, 
                    seed_worker=seed_worker
                )

                es_results = model.es_train(train_loader, val_loader, **kwargs)

                # Unpack results
                best_train_loss, best_train_acc, best_val_loss, best_val_acc, learning_curve = es_results
                results = {
                    'final_train_loss': best_train_loss,
                    'final_train_acc': best_train_acc,
                    'final_val_loss': best_val_loss,
                    'final_val_acc': best_val_acc,
                    'learning_curve_es': learning_curve
                }
                candidate.metrics["es_results"] = results


    # def validate_generation(self, X_val, y_val, metric='val'):

    #     active_individuals = self.candidates.keys()
    #     for i in active_individuals:
    #         candidate = self.candidates[i]
    #         model = candidate.model
    #         batch_size = candidate.batch_size
    #         seed = candidate.architecture.get("seed", None)
    #         g, seed_worker = set_seed(seed)
            
    #         # Create a DataLoader with the architecture-specific batch size
    #         val_loader = dp.create_dataloader(X=X_val, 
    #                                             y=y_val, 
    #                    batch_size=batch_size, 
    #                    generator=g, 
    #                    seed_worker=seed_worker)
    #         val_loss, val_acc = model.evaluate(val_loader)

    #         candidate.log_metric(metric, 'loss', val_loss)
    #         candidate.log_metric(metric, 'acc', val_acc)


    def spawn_new_candidates(self, search_space):
        self.size = len(self.candidates)
        n_new = self.max_individuals - self.size
        n_basic = n_new // 2
        n_adv = n_new - n_basic

        rng = self.repro.np_rng

        new_generation = {}
        for _ in range(n_basic):
            arch = search_space.sample_architecture(rng=rng)
            model = search_space.create_model(arch, task_type=self.task_type)
            new_generation[self.individuals_created] = Candidate(model, arch, starting_instances=self.starting_instances, id_counter=self.individuals_created + 1)
            self.individuals_created += 1

        for _ in range(n_adv):
            child_arch = breed_and_mutate(self.candidates, rng=rng)  # see next step
            child_model = search_space.create_model(child_arch, task_type=self.task_type)
            new_generation[self.individuals_created] = Candidate(child_model, child_arch, id_counter=self.individuals_created + 1, starting_instances=self.starting_instances)
            self.individuals_created += 1

        self.candidates.update(new_generation)
        self.size = len(self.candidates)



    def prune_candidates(self,
                        baseline_metric,
                        base_drop=0.1,   # gentler start
                        max_drop=0.3,   # softer ceiling
                        elite_fraction=0.2,  # bigger elite buffer
                        incubator_fraction=0.1,
                        min_survivors=5):
        """
        Softer unified pruning for better forecast stability.
        - Does not prune aggressively until candidates have at least 3 val points.
        - Larger elite buffer and optional incubator pool.
        """

        n = len(self.candidates)
        if n <= min_survivors:
            self.worst_individuals = []
            print(f"[Prune] total={n}, below survivor floor, no pruning.")
            return

        b_ref = max(c.n_instances[-1] for c in self.candidates.values())

        hopeless, survivors = [], []
        for k, cand in self.candidates.items():
            # convex LB only if enough val points exist
            if len(cand.get_metric("val", "acc")) >= 3 and convex_lb_discard(cand, baseline_metric, b_ref):
                hopeless.append(k)
            else:
                survivors.append(k)

        # If too few val points, skip pruning entirely
        if all(len(c.get_metric("val", "acc")) < 3 for c in self.candidates.values()):
            self.worst_individuals = []
            print(f"[Prune] Skipped pruning, not enough anchors yet.")
            return

        # Drop fraction
        frac = min(max_drop, base_drop + 0.01 * self.generations_completed)
        n_drop = int(len(survivors) * frac)

        # --- Rank by adjusted probability (boost by training effort) ---
        ranked_by_prob = sorted(
            [
                (
                    k,
                    self.candidates[k].metrics.get("p_above_goal", 0.0)
                    * (1.0 + 0.2 * self.candidates[k].batches_trained)  # boost per batch
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

        # Incubator buffer (keep some randoms alive)
        incubator_count = max(1, int(incubator_fraction * n))
        incubators = {k for k, _ in ranked_by_prob[-incubator_count:]}  # from the bottom

        # Survivors by prob
        n_keep = max(min_survivors, len(survivors) - n_drop)
        keep_prob = {k for k, _ in ranked_by_prob[:n_keep]}

        keep = keep_prob | elites | incubators
        worst = [k for k in self.candidates if k not in keep] + hopeless
        worst = list(set(worst))

        self.worst_individuals = worst

        print(f"[Prune] total={n}, hopeless={len(hopeless)}, "
            f"dropped={len(worst)}, kept={n - len(worst)} ")

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
                   inject_fraction=0.2,
                   base_drop=0.2,
                   max_drop=0.5,
                   track_all_models=False):

        log = {"gen": self.generations_completed + 1}

        # --- Phase 1: Train ---
        before_train = len(self.candidates)
        self.train_generation(X_train, y_train,
                            training_mode="oe",
                            X_val=X_val, y_val=y_val)
        log["trained"] = before_train

        # --- Compute estimate (wall-time) ---
        total_epoch_times = []
        for cand in self.candidates.values():
            if "epoch_time" in cand.metrics:
                total_epoch_times.extend(cand.metrics["epoch_time"])
        log["compute_this_gen"] = float(sum(total_epoch_times)) if total_epoch_times else 0.0
        # --- Phase 2: Forecast + scoring ---
        forecast_generation(self.candidates, dataset_size=len(X_train),
                            min_val_points=3, growth=1.4, extra_full_passes=3)
        check_higher_than_baseline(self.candidates, baseline_metric)
        dynamic_goal = max(
            baseline_metric or 0.0,
            max((c.get_metric("val", "acc", last_only=True) or 0.0) for c in self.candidates.values()),
            max((c.metrics.get("forecasted_val_acc", 0.0) or 0.0) for c in self.candidates.values())
        )
        compute_and_log_p_above_goal(self.candidates, goal_metric=dynamic_goal)
        score_individuals(self.candidates)

        # --- Phase 3: Unified pruning (includes convex LB + hybrid drop) ---
        survivors_before = len(self.candidates)
        self.prune_candidates(baseline_metric=dynamic_goal,
                            base_drop=base_drop,
                            max_drop=max_drop,
                            elite_fraction=0.1,
                            min_survivors=5)
        self.drop_worst_individuals()
        survivors_after = len(self.candidates)

        log["hybrid_dropped"] = survivors_before - survivors_after
        log["survivors"] = survivors_after

        # --- Phase 4: Exploration via trickle spawn ---
        before_spawn = len(self.candidates)
        self.spawn_new_candidates(self.search_space)
        after_spawn = len(self.candidates)
        log["spawned"] = after_spawn - before_spawn
        log["final_population"] = after_spawn

        # --- Ledger update ---
        self.size = len(self.candidates)
        self.current_snapshot = self.build_ledger()
        if track_all_models:
            self.cumulative_ledger = (
                pd.concat([self.cumulative_ledger, self.current_snapshot])
                .drop_duplicates(subset="id", keep="last")
            )

        # Save the log
        prev_total = self.generation_logs[-1]["cumulative_compute"] if self.generation_logs else 0.0
        log["cumulative_compute"] = prev_total + log["compute_this_gen"]

        self.generation_logs.append(log)


        return self.candidates

    

    def run_ebe(self,
            X_train, y_train, X_val, y_val,
            baseline_metric,
            max_generations=20,
            time_budget=60,
            inject_fraction=0.2,
            base_drop=0.2,
            max_drop=0.5,
            track_all_models=False):
        """
        Full EBE loop:
        - Repeatedly run packaged generations with convex LB + hybrid pruning + trickle spawn.
        - Stop if time budget is exceeded or max_generations reached.
        - At the end, compute expected utility decision.
        """

        self.current_snapshot = self.initial_ledger
        self.cumulative_ledger = self.initial_ledger if track_all_models else None

        start_time = time.time()

        for gen in range(max_generations):
            elapsed = time.time() - start_time
            if elapsed >= time_budget:
                print(f"[EBE] Time budget exceeded at gen {gen+1}, elapsed={elapsed:.2f}s")
                break

            print(f"\n=== Generation {gen+1} ===")

            self.candidates = self.run_generation(
                X_train, y_train, X_val, y_val,
                baseline_metric=baseline_metric,
                inject_fraction=inject_fraction,
                base_drop=base_drop,
                max_drop=max_drop,
                track_all_models=track_all_models
            )

            self.generations_completed += 1

        # --- Final decision ---
        self.decision, self.eu, self.p, self.benefit, self.cost = self.worth_training_neural_bayes(
            baseline_metric=baseline_metric,
            target_epochs=100,
            cost_scale=1e-3
        )
        print("EBE process completed.")

        print("Training top by ES (fidelity check)")
        self.fidelity_ledger = self.es_fidelity_from_ledger(
            ledger_df=self.current_snapshot,   # <-- the latest ledger DataFrame
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            es_patience=30,
            top_fraction=0.2
        )

        return self.current_snapshot



    def build_ledger(self, export_as='pandas'):
        current_candidates = []
        active_individuals = self.candidates.keys()
        for i in active_individuals:
            candidate = self.candidates[i]
            current_candidates.append(candidate.build_dict())           
        
        if export_as == 'pandas':
            df = pd.DataFrame(current_candidates).sort_values(by='score', ascending=False)
            return df.copy(deep=True)
        elif export_as == 'json':
            return json.dumps(current_candidates, indent=4)


    def worth_training_neural_bayes(self, baseline_metric, target_epochs=10, cost_scale=1.0):
        """
        Simple cost model: projected_remaining_time = epoch_time * (target_epochs - epochs_trained)
        cost = cost_scale * projected_remaining_time
        """
        best_cand = max(self.candidates.values(),
                        key=lambda c: c.metrics.get("p_above_goal", 0.0))

        p     = best_cand.metrics.get("p_above_goal", 0.0)
        fcst  = best_cand.metrics.get("forecasted_val_acc", 0.0)
        benefit = max(0.0, fcst - baseline_metric)

        epochs_done = getattr(best_cand, "epochs_trained", 0)
        remaining   = max(0, target_epochs - epochs_done)

        # avg epoch time across the quick passes we’ve run (you have 1 right now, still fine)
        epoch_times = best_cand.metrics.get("epoch_time", [])
        avg_epoch_time = float(sum(epoch_times) / len(epoch_times)) if epoch_times else 0.0

        projected_remaining_time = avg_epoch_time * remaining
        cost = cost_scale * projected_remaining_time

        EU = p * benefit - (1 - p) * cost
        return EU > 0, EU, p, benefit, cost

    

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
    

    def es_fidelity_from_ledger(self, ledger_df, 
                            X_train, y_train, X_val, y_val,
                            es_patience=30, top_fraction=0.2):
        """
        Rebuild models from ledger rows and retrain them with ES
        to compare forecast vs actual performance.
        """

        import pandas as pd
        from architecture_generator import create_model_from_row

        # rank by forecasted_val_acc
        ranked = ledger_df.sort_values("forecasted_val_acc", ascending=False)
        n_keep = max(1, int(len(ranked) * top_fraction))
        chosen = ranked.head(n_keep)

        fidelity_records = []

        for _, row in chosen.iterrows():
            # rebuild model from the ledger row
            model = create_model_from_row(
                row,
                input_size=self.search_space.input_size,
                output_size=self.search_space.output_size,
                task_type=self.task_type
            )
            cand = Candidate(model, row.to_dict(), starting_instances=row["n_instances"][0], id_counter=row["id"])

            # copy over forecast info
            cand.metrics["forecasted_val_acc"] = row["forecasted_val_acc"]
            cand.metrics["forecast_horizon_time"] = row.get("forecast_horizon_time")

            # loaders
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

            # train with ES
            results = model.es_train(
                candidate=cand,
                train_loader=train_loader,
                val_loader=val_loader,
                es_patience=es_patience,
                verbose=True,
                task_type=self.task_type,
                return_lc=True
            )

            best_train_loss, best_train_acc, best_val_loss, best_val_acc, lc = results

            fidelity_records.append({
                "id": cand.id,
                "forecasted_val_acc": cand.metrics["forecasted_val_acc"],
                "forecast_horizon_time": cand.metrics.get("forecast_horizon_time"),
                "fidelity_val_acc": best_val_acc,
                "fidelity_train_acc": best_train_acc,
                "fidelity_val_loss": best_val_loss,
                "learning_curve": lc
            })
        
        for rec in fidelity_records:
            print("DEBUG fidelity_record:")
            for k, v in rec.items():
                print(f"  {k}: {type(v)} -> {v if not isinstance(v, list) else f'list(len={len(v)})'}")


        fidelity_ledger = pd.DataFrame(fidelity_records).sort_values("fidelity_val_acc", ascending=False)
        self.fidelity_ledger = fidelity_ledger
        return fidelity_ledger
    
    def compare_forecast_vs_fidelity(self):
        """
        Compare forecasted validation accuracy against actual ES accuracy.
        Returns error stats (MAE, bias) and per-candidate deltas.
        """
        if not hasattr(self, "fidelity_ledger"):
            raise RuntimeError("Run post_ebe_fidelity_check or es_fidelity_from_ledger first.")

        df = self.fidelity_ledger.copy()

        fcst = df["forecasted_val_acc"].astype(float)
        actual = df["fidelity_val_acc"].astype(float)

        deltas = actual - fcst
        mae = float(np.mean(np.abs(deltas)))
        bias = float(np.mean(deltas))

        report = {
            "MAE": mae,
            "Bias": bias,
            "n": len(df),
            "details": pd.DataFrame({
                "id": df["id"],
                "forecasted_val_acc": fcst,
                "fidelity_val_acc": actual,
                "delta": deltas
            }).sort_values("delta", ascending=False)
        }

        return report
