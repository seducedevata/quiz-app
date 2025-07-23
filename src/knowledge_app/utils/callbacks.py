from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


class StopTrainingCallback(TrainerCallback):
    def __init__(self, app_instance):
        super().__init__()
        self.app_instance = app_instance

    def on_step_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        if getattr(self.app_instance, "should_stop_training", False):
            import logging
            logger = logging.getLogger(__name__)
            logger.info("Stop training signal received by callback. Stopping training.")
            control.should_training_stop = True


class ProgressUpdateCallback(TrainerCallback):
    def __init__(self, app_instance, total_steps=3000):
        super().__init__()
        self.app_instance = app_instance
        self.total_steps = total_steps
        self.last_step = 0
        self.start_time = None
        self.step_times = []
        self.last_update_time = 0
        self.update_interval = 1.0  # Update UI every 1 second
        self.progress_ranges = {
            "preprocessing": (0, 20),
            "training": (20, 95),
            "finalization": (95, 100),
        }

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        import time
        import logging
        logger = logging.getLogger(__name__)

        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.step_times = []
        logger.info(
            f"Training started at {time.strftime('%H:%M:%S', time.localtime(self.start_time))}"
        )

        # Initial progress update
        self._emit_progress(
            step=0,
            total_steps=self.total_steps,
            stage="training",
            extra_info="Initializing training...",
        )

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        import time

        current_time = time.time()

        # Update progress at regular intervals or on significant progress
        if (
            (current_time - self.last_update_time >= self.update_interval)
            or (state.global_step % 50 == 0)
            or (state.global_step == self.total_steps)
        ):

            # Store the current step for resuming progress tracking
            self.last_step = state.global_step

            # Update step times for ETA calculation (keep last 20 steps)
            if len(self.step_times) >= 20:
                self.step_times.pop(0)
            self.step_times.append(current_time)

            # Calculate progress within training phase (20-95%)
            base_progress = self.progress_ranges["training"][0]
            progress_range = self.progress_ranges["training"][1] - base_progress
            training_progress = (state.global_step / self.total_steps) * progress_range
            progress_percentage = min(
                self.progress_ranges["training"][1], base_progress + training_progress
            )

            # Calculate ETA
            eta_str = self._calculate_eta(state.global_step)
            elapsed_str = self._format_time(current_time - self.start_time)

            # Construct status message
            status_msg = (
                f"Training in progress... Step {state.global_step}/{self.total_steps} "
                f"({progress_percentage:.1f}%)\n"
                f"Elapsed: {elapsed_str} | ETA: {eta_str}"
            )

            # Calculate speed
            if len(self.step_times) >= 2:
                steps_per_sec = (len(self.step_times) - 1) / (
                    self.step_times[-1] - self.step_times[0]
                )
                status_msg += f" | {steps_per_sec:.1f} steps/sec"

            # Emit progress update
            self._emit_progress(
                step=state.global_step,
                total_steps=self.total_steps,
                stage="training",
                extra_info=status_msg,
            )

            # Update last update time
            self.last_update_time = current_time

    def _calculate_eta(self, current_step):
        """Calculate estimated time remaining."""
        if len(self.step_times) >= 2:
            avg_time_per_step = (self.step_times[-1] - self.step_times[0]) / (
                len(self.step_times) - 1
            )
            steps_remaining = self.total_steps - current_step
            eta_seconds = avg_time_per_step * steps_remaining
            return self._format_time(eta_seconds)
        return "Calculating..."

    def _format_time(self, seconds):
        """Format time in seconds to human readable string."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            seconds = int(seconds % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"

    def _emit_progress(self, step, total_steps, stage, extra_info=""):
        """Emit progress update with consistent formatting."""
        # Get progress range for current stage
        start_progress, end_progress = self.progress_ranges.get(stage, (0, 100))

        # Calculate progress within range
        if total_steps > 0:
            stage_progress = (step / total_steps) * (end_progress - start_progress)
            progress = min(end_progress, start_progress + stage_progress)
        else:
            progress = start_progress

        # Emit progress
        if (
            hasattr(self.app_instance, "settings_menu")
            and self.app_instance.settings_menu is not None
        ):
            if hasattr(self.app_instance.settings_menu, "training_progress_bar"):
                self.app_instance.settings_menu.training_progress_bar.setVisible(True)
                self.app_instance.settings_menu.training_progress_bar.setValue(int(progress))

            if hasattr(self.app_instance.settings_menu, "training_status_label"):
                self.app_instance.settings_menu.training_status_label.setText(
                    f"Status: {extra_info}"
                )

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        """Handle training completion."""
        self._emit_progress(
            step=self.total_steps,
            total_steps=self.total_steps,
            stage="finalization",
            extra_info=f"Training completed successfully! (Total steps: {state.global_step})",
        )