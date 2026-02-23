import copy

import torch
import torch.optim as optim
try:
    from torch.amp import GradScaler, autocast

    def _autocast_context(enabled, device):
        return autocast("cuda", enabled=(enabled and device != "cpu"))

except ImportError:
    from torch.cuda.amp import GradScaler, autocast

    def _autocast_context(enabled, device):
        return autocast(enabled=(enabled and device != "cpu"))
from sklearn.model_selection import train_test_split


class FlowMatchingTrainer:
    def __init__(
            self,
            model,
            data,
            data_errors=None,
            data_observed_mask=None,
            val_data=None,
            val_errors=None,
            val_observed_mask=None,
            condition_mask_generator=None,
            batch_size=128,
            lr=1e-3,
            sigma_min=0.001,
            time_prior_exponent=1.0,
            inner_train_loop_size=1000,
            early_stopping_patience=20,
            val_split=0.15,
            val_every_n_epochs=1,
            val_num_batches=200,
            dense_ratio=0.8,
            use_amp=False,
            grad_clip_norm=1.0,
            weight_decay=1e-4,
            use_wandb=True,
            wandb_project="flow-matching",
            wandb_config=None,
            wandb_run_name=None,
            checkpoint_path=None,
            device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.condition_mask_generator = condition_mask_generator
        self.batch_size = batch_size
        self.lr = lr
        self.sigma_min = sigma_min
        self.time_prior_exponent = time_prior_exponent
        self.inner_train_loop_size = inner_train_loop_size
        self.early_stopping_patience = early_stopping_patience
        self.val_every_n_epochs = val_every_n_epochs
        self.val_num_batches = val_num_batches
        self.dense_ratio = dense_ratio
        self.use_amp = use_amp
        self.grad_clip_norm = grad_clip_norm
        self.use_wandb = use_wandb
        self._wandb = None
        self.checkpoint_path = checkpoint_path
        self.device = device

        # AMP scaler for mixed precision training
        try:
            self.scaler = GradScaler('cuda', enabled=(use_amp and device != 'cpu'))
        except TypeError:
            self.scaler = GradScaler(enabled=(use_amp and device != 'cpu'))

        # Whether to use pinned memory (only useful for GPU training)
        use_pin = (device != 'cpu' and torch.cuda.is_available())

        if val_data is not None:
            # Pre-split data provided — use directly (no internal split)
            self.train_data = self._to_cpu_tensor(data, pin=use_pin)
            self.val_data = self._to_cpu_tensor(val_data, pin=use_pin)
            self.train_errors = self._to_cpu_tensor(data_errors, pin=use_pin)
            self.val_errors = self._to_cpu_tensor(val_errors, pin=use_pin)
            self.train_observed = self._to_cpu_tensor(data_observed_mask, pin=use_pin)
            self.val_observed = self._to_cpu_tensor(val_observed_mask, pin=use_pin)
        elif val_split > 0:
            # Legacy path: split internally (kept for backward compatibility)
            aux_arrays = []
            has_errors = data_errors is not None
            has_observed = data_observed_mask is not None
            if has_errors:
                aux_arrays.append(data_errors)
            if has_observed:
                aux_arrays.append(data_observed_mask)

            if aux_arrays:
                splits = train_test_split(data, *aux_arrays, test_size=val_split, random_state=42)
                train_data_s, val_data_s = splits[0], splits[1]
                idx = 2
                if has_errors:
                    train_errors, val_errors_s = splits[idx], splits[idx + 1]
                    idx += 2
                else:
                    train_errors = val_errors_s = None
                if has_observed:
                    train_observed, val_observed_s = splits[idx], splits[idx + 1]
                else:
                    train_observed = val_observed_s = None
            else:
                train_data_s, val_data_s = train_test_split(data, test_size=val_split, random_state=42)
                train_errors = val_errors_s = None
                train_observed = val_observed_s = None

            self.train_data = self._to_cpu_tensor(train_data_s, pin=use_pin)
            self.val_data = self._to_cpu_tensor(val_data_s, pin=use_pin)
            self.train_errors = self._to_cpu_tensor(train_errors, pin=use_pin)
            self.val_errors = self._to_cpu_tensor(val_errors_s, pin=use_pin)
            self.train_observed = self._to_cpu_tensor(train_observed, pin=use_pin)
            self.val_observed = self._to_cpu_tensor(val_observed_s, pin=use_pin)
        else:
            self.train_data = self._to_cpu_tensor(data, pin=use_pin)
            self.val_data = None
            self.train_errors = self._to_cpu_tensor(data_errors, pin=use_pin) if data_errors is not None else None
            self.val_errors = None
            self.train_observed = self._to_cpu_tensor(data_observed_mask, pin=use_pin) if data_observed_mask is not None else None
            self.val_observed = None

        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        # Epoch indices for cap-based curriculum sampling (set via epoch callback)
        self._train_epoch_indices = None
        self._train_epoch_weights = None
        self._train_idx_pos = 0
        self._val_epoch_indices = None
        self._val_epoch_weights = None
        self._val_idx_pos = 0
        self.best_model_state = None

        self.num_nodes = self.train_data.shape[1]
        self.node_ids = torch.arange(self.num_nodes).unsqueeze(0).repeat(self.batch_size, 1).to(device)

        # Initialize wandb (lazy import — only needed when use_wandb=True)
        if self.use_wandb:
            import wandb
            self._wandb = wandb
            config = wandb_config or {}
            config.update({
                "batch_size": batch_size,
                "lr": lr,
                "sigma_min": sigma_min,
                "time_prior_exponent": time_prior_exponent,
                "inner_train_loop_size": inner_train_loop_size,
                "num_nodes": self.num_nodes,
                "train_samples": len(self.train_data),
                "val_samples": len(self.val_data) if self.val_data is not None else 0,
                "has_errors": self.train_errors is not None,
                "has_observed_mask": self.train_observed is not None,
                "val_num_batches": val_num_batches,
            })
            self._wandb.init(project=wandb_project, name=wandb_run_name, config=config)
            self._wandb.watch(self.model, log="all", log_freq=100)

    @staticmethod
    def _to_cpu_tensor(arr, pin=False):
        """Convert numpy array to CPU tensor, optionally with pinned memory."""
        if arr is None:
            return None
        t = torch.tensor(arr, dtype=torch.float32)
        if pin:
            t = t.pin_memory()
        return t

    def set_epoch_indices(self, indices, sample_weights=None):
        """Set pre-computed training indices for this epoch.

        Args:
            indices: 1-D numpy array of global star indices (shuffled),
                     produced by build_epoch_indices().  Each star appears
                     at most once.  Epoch length = len(indices) // batch_size.
            sample_weights: Optional 1-D numpy array aligned with ``indices``.
        """
        self._train_epoch_indices = torch.from_numpy(indices).long()
        if sample_weights is not None:
            if len(sample_weights) != len(indices):
                raise ValueError(
                    f"sample_weights length ({len(sample_weights)}) must match indices length ({len(indices)})"
                )
            self._train_epoch_weights = torch.from_numpy(sample_weights).float()
        else:
            self._train_epoch_weights = None
        self._train_idx_pos = 0

    def set_val_epoch_indices(self, indices, sample_weights=None):
        """Set pre-computed validation indices for this epoch.

        Args:
            indices: 1-D numpy array of global star indices (shuffled).
            sample_weights: Optional 1-D numpy array aligned with ``indices``.
        """
        self._val_epoch_indices = torch.from_numpy(indices).long()
        if sample_weights is not None:
            if len(sample_weights) != len(indices):
                raise ValueError(
                    f"sample_weights length ({len(sample_weights)}) must match indices length ({len(indices)})"
                )
            self._val_epoch_weights = torch.from_numpy(sample_weights).float()
        else:
            self._val_epoch_weights = None
        self._val_idx_pos = 0

    def sample_batch(self, from_val=False):
        """Return the next batch by advancing through pre-built epoch indices.

        If epoch indices are set (via set_epoch_indices / set_val_epoch_indices),
        returns the next contiguous slice.  Otherwise falls back to uniform
        random sampling (no curriculum).

        Data is stored on CPU and transferred to GPU per-batch with
        non_blocking=True.
        """
        if from_val:
            data, errors, observed = self.val_data, self.val_errors, self.val_observed
            indices, pos, weights = self._val_epoch_indices, self._val_idx_pos, self._val_epoch_weights
        else:
            data, errors, observed = self.train_data, self.train_errors, self.train_observed
            indices, pos, weights = self._train_epoch_indices, self._train_idx_pos, self._train_epoch_weights

        if indices is not None and pos < len(indices):
            idx = indices[pos : pos + self.batch_size]
            batch_weights = weights[pos : pos + self.batch_size] if weights is not None else None
            # Advance position counter
            if from_val:
                self._val_idx_pos += len(idx)
            else:
                self._train_idx_pos += len(idx)
        else:
            # Fallback: uniform random sampling (no curriculum active)
            idx = torch.randint(0, data.shape[0], (self.batch_size,))
            batch_weights = None

        batch_data = data[idx].to(self.device, non_blocking=True)
        batch_errors = errors[idx].to(self.device, non_blocking=True) if errors is not None else None
        batch_observed = observed[idx].to(self.device, non_blocking=True) if observed is not None else None
        if batch_weights is None:
            batch_weights = torch.ones(len(idx), dtype=torch.float32)
        batch_weights = batch_weights.to(self.device, non_blocking=True)
        return batch_data, batch_errors, batch_observed, batch_weights

    def build_edge_mask(self, dense_ratio=0.7, observed_mask=None):
        N = self.num_nodes
        n_dense = int(self.batch_size * dense_ratio)
        n_sparse = self.batch_size - n_dense
        dev = self.device

        dense = torch.ones((n_dense, N, N), dtype=torch.bool, device=dev)

        if n_sparse > 0:
            # Vectorized: outer AND product of per-node keep masks
            keep = torch.rand(n_sparse, N, device=dev) >= 0.2             # (n_sparse, N)
            sparse = keep.unsqueeze(1) & keep.unsqueeze(2)                 # (n_sparse, N, N)
            diag_idx = torch.arange(N, device=dev)
            sparse[:, diag_idx, diag_idx] = True                           # restore self-loops
            masks = torch.cat([dense, sparse], dim=0)
        else:
            masks = dense

        indices = torch.randint(0, masks.shape[0], (self.batch_size,), device=dev)
        masks = masks[indices]

        # Mask out unobserved nodes: no attention to/from them
        if observed_mask is not None:
            obs = observed_mask.squeeze(-1).bool() if observed_mask.dim() == 3 else observed_mask.bool()
            obs_edge = obs.unsqueeze(1) & obs.unsqueeze(2)  # (B, N, N)
            masks = masks & obs_edge
            # Restore self-loops so softmax always has at least one valid entry
            diag_idx = torch.arange(N, device=dev)
            masks[:, diag_idx, diag_idx] = True

        return masks

    def sample_t(self, batch_size):
        t = torch.rand(batch_size, device=self.device)
        return t.pow(1 / (1 + self.time_prior_exponent))

    def flow_matching_loss(
        self,
        x0,
        x1,
        node_ids,
        condition_mask,
        edge_mask,
        x_errors=None,
        observed_mask=None,
        sample_weights=None,
    ):
        """Compute standard flow matching loss."""
        B, N = x0.shape
        t = self.sample_t(B).reshape(B, 1, 1)
        k = (1.0 - self.sigma_min)

        # Geodesic mix + clamp to non-free values (no grad into x1 on those dims)
        xt = (1 - k * t) * x0.unsqueeze(-1) + t * x1.unsqueeze(-1)

        # free = 1 only for dims that are NOT conditioned AND are observed
        cond = condition_mask.to(x0.dtype).unsqueeze(-1)  # [B,D,1]
        free = 1.0 - cond
        if observed_mask is not None:
            obs = observed_mask.to(x0.dtype)
            if obs.dim() == 2:
                obs = obs.unsqueeze(-1)  # [B,D,1]
            free = free * obs  # exclude unobserved dims from free subspace

        # Clamp non-free dims (conditioned OR unobserved) to x1 with no grad
        xt = xt * free + x1.unsqueeze(-1).detach() * (1.0 - free)

        # predict velocity
        v_pred = self.model(t, xt, node_ids, condition_mask, edge_mask, x_errors, observed_mask=observed_mask).squeeze(-1)  # [B,D]
        v_tgt = (x1 - k * x0)  # [B,D]

        # Project BOTH velocities to the free subspace (stronger than masking the loss)
        free_s = free.squeeze(-1)  # [B,D]
        v_pred = v_pred * free_s
        v_tgt = v_tgt * free_s

        # MSE over free dims, normalized per sample by #free dims
        diff = (v_pred - v_tgt).pow(2)  # [B,D]
        counts = free_s.sum(-1).clamp_min(1.0)  # [B]
        per_sample = diff.sum(-1) / counts
        if sample_weights is None:
            loss = per_sample.mean()
        else:
            weights = sample_weights.to(per_sample.dtype).reshape(-1)
            denom = weights.sum().clamp_min(1e-8)
            loss = (per_sample * weights).sum() / denom
        return loss

    def training_step(
        self,
        x0,
        x1,
        node_ids,
        condition_mask,
        edge_mask,
        x_errors,
        observed_mask=None,
        sample_weights=None,
    ):
        """Perform one training step with optional mixed-precision."""
        with _autocast_context(self.use_amp, self.device):
            loss = self.flow_matching_loss(
                x0,
                x1,
                node_ids,
                condition_mask,
                edge_mask,
                x_errors,
                observed_mask=observed_mask,
                sample_weights=sample_weights,
            )

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {'loss': loss.item()}

    @torch.no_grad()
    def validate(self, num_batches=None):
        """Compute validation loss over multiple batches.

        When epoch indices are set (via set_val_epoch_indices), iterates
        through them once.  Otherwise falls back to num_batches random
        batches.
        """
        if self.val_data is None:
            return None

        # Determine number of validation steps
        if self._val_epoch_indices is not None and num_batches is None:
            n_steps = len(self._val_epoch_indices) // self.batch_size
            self._val_idx_pos = 0   # reset to iterate from the start
        else:
            n_steps = num_batches or self.val_num_batches

        self.model.eval()
        total_val_loss = 0.0

        for _ in range(n_steps):
            x1, x_errors, x_observed, x_weights = self.sample_batch(from_val=True)
            x0 = torch.randn_like(x1)
            condition_mask = next(self.condition_mask_generator)
            # Never condition on unobserved nodes
            if x_observed is not None:
                condition_mask = condition_mask * x_observed
            # Same edge mask distribution as training so losses are comparable.
            # (At inference time, dense masks are used independently.)
            edge_mask = self.build_edge_mask(dense_ratio=self.dense_ratio, observed_mask=x_observed)

            with _autocast_context(self.use_amp, self.device):
                loss = self.flow_matching_loss(
                    x0,
                    x1,
                    self.node_ids,
                    condition_mask,
                    edge_mask,
                    x_errors,
                    observed_mask=x_observed,
                    sample_weights=x_weights,
                )
            total_val_loss += loss.item()

        self.model.train()
        return total_val_loss / max(n_steps, 1)

    def fit(self, epochs=100, verbose=True, epoch_callback=None,
            post_epoch_callback=None, lr_scheduler=None):
        """Train the model.

        Args:
            epochs: Number of training epochs.
            verbose: Print progress.
            epoch_callback: Optional callable(trainer, epoch, total_epochs)
                called at the start of each epoch.  Expected to call
                set_epoch_indices / set_val_epoch_indices to define the
                data seen this epoch.
            post_epoch_callback: Optional callable(trainer, epoch, total_epochs)
                called after training steps, validation, and logging, but
                before early stopping check.  Useful for secondary metrics
                (e.g., validation at τ=τ_max).
            lr_scheduler: Optional LR scheduler, stepped at the end of each epoch
                (after all optimizer steps).
        """
        best_loss = float("inf")
        no_improve = 0
        global_step = 0

        for epoch in range(epochs):
            if epoch_callback is not None:
                epoch_callback(self, epoch, epochs)
            self.model.train()
            total_loss = 0.0

            # Epoch length: determined by pre-built indices, or fallback
            if self._train_epoch_indices is not None:
                n_steps = len(self._train_epoch_indices) // self.batch_size
            else:
                n_steps = self.inner_train_loop_size

            # Prefetch first batch so CPU→GPU transfer overlaps with compute
            next_batch = self.sample_batch(from_val=False)
            for step in range(n_steps):
                x1, x_errors, x_observed, x_weights = next_batch
                # Prefetch next batch while GPU processes current one
                if step < n_steps - 1:
                    next_batch = self.sample_batch(from_val=False)

                x0 = torch.randn_like(x1)
                condition_mask = next(self.condition_mask_generator)
                # Never condition on unobserved nodes
                if x_observed is not None:
                    condition_mask = condition_mask * x_observed
                edge_mask = self.build_edge_mask(dense_ratio=self.dense_ratio, observed_mask=x_observed)

                step_metrics = self.training_step(
                    x0,
                    x1,
                    self.node_ids,
                    condition_mask,
                    edge_mask,
                    x_errors,
                    observed_mask=x_observed,
                    sample_weights=x_weights,
                )

                total_loss += step_metrics['loss']
                global_step += 1

                # Log to wandb every N steps
                if self.use_wandb and step % 100 == 0:
                    self._wandb.log({
                        "train_loss_step": step_metrics['loss'],
                        "epoch": epoch + 1,
                        "step": global_step
                    })

            avg_train_loss = total_loss / max(n_steps, 1)

            # Validate periodically
            val_loss = None
            if epoch % self.val_every_n_epochs == 0:
                val_loss = self.validate()

            # Logging
            if verbose:
                log_str = f"Epoch {epoch + 1}: train_loss = {avg_train_loss:.6f}"
                if val_loss is not None:
                    log_str += f", val_loss = {val_loss:.6f}"
                print(log_str)

            if self.use_wandb:
                log_dict = {
                    "train_loss_epoch": avg_train_loss,
                    "epoch": epoch + 1
                }
                if val_loss is not None:
                    log_dict["val_loss"] = val_loss
                self._wandb.log(log_dict)

            # Post-epoch callback (e.g., tau_max validation)
            if post_epoch_callback is not None:
                post_epoch_callback(self, epoch, epochs)

            # Step LR scheduler (after optimizer steps to avoid PyTorch warning)
            if lr_scheduler is not None:
                lr_scheduler.step()

            # Early stopping based on validation loss (if available) or training loss
            monitor_loss = val_loss if val_loss is not None else avg_train_loss

            if monitor_loss < best_loss:
                best_loss = monitor_loss
                no_improve = 0
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                if self.checkpoint_path is not None:
                    torch.save(self.best_model_state, self.checkpoint_path)
                if self.use_wandb:
                    self._wandb.run.summary["best_loss"] = best_loss
                    self._wandb.run.summary["best_epoch"] = epoch + 1
            else:
                no_improve += 1

            if no_improve >= self.early_stopping_patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

        # Load best model
        self.model.load_state_dict(self.best_model_state)

        if self.use_wandb:
            self._wandb.unwatch(self.model)
            self._wandb.finish()

        return self.model
