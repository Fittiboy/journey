import ipywidgets as widgets
from IPython.display import display
import time
import threading

class StopWatch:
    def __init__(self):
        # Internal time-tracking
        self.start_time = None
        self.elapsed = 0.0
        self.running = False

        # Create UI elements
        self.label = widgets.Label(value="0:00.00")
        self.start_button = widgets.Button(
            description="▶",
            button_style='success',
            layout=widgets.Layout(width='40px')
        )
        self.stop_button = widgets.Button(
            description="■",
            button_style='warning',
            layout=widgets.Layout(width='40px')
        )
        self.reset_button = widgets.Button(
            description="↺",
            button_style='info',
            layout=widgets.Layout(width='40px')
        )

        # Wire buttons to event handlers
        self.start_button.on_click(self.on_start)
        self.stop_button.on_click(self.on_stop)
        self.reset_button.on_click(self.on_reset)
        
        # Assemble the widget layout
        self.box = widgets.HBox([
            self.label,
            self.start_button,
            self.stop_button,
            self.reset_button
        ])

    def on_start(self, _):
        """Start the stopwatch."""
        if not self.running:
            self.running = True
            # Set the start time to 'now' minus what we've already counted
            self.start_time = time.time() - self.elapsed
            # Run the updating in a separate thread so it doesn’t block the notebook
            self._update_thread = threading.Thread(target=self._update_time)
            self._update_thread.start()

    def on_stop(self, _):
        """Stop the stopwatch."""
        if self.running:
            self.running = False
            # Calculate final elapsed time
            self.elapsed = time.time() - self.start_time

    def on_reset(self, _):
        """Reset the stopwatch to zero."""
        # Only reset if we're not running
        if not self.running:
            self.elapsed = 0.0
            self.label.value = "0:00.00"

    def _update_time(self):
        """Internal helper to update elapsed time periodically."""
        while self.running:
            current = time.time() - self.start_time
            # Convert to H:MM:SS.ss if needed
            hours = int(current) // 3600
            minutes = (int(current) % 3600) // 60
            seconds = current % 60
            if hours > 0:
                self.label.value = f"{hours}:{minutes:02}:{seconds:05.2f}"
            else:
                self.label.value = f"{minutes}:{seconds:05.2f}"
            time.sleep(0.1)

    def display(self):
        """Display the stopwatch widget in the notebook."""
        display(self.box)

# Create and display the stopwatch
sw = StopWatch()
sw.display()