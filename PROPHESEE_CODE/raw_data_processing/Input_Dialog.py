import tkinter as tk
from tkinter import  messagebox
class ThresholdDialog:
    def __init__(self, master):
        self.master = master
        self.result = None
        self.selected_value = tk.StringVar(value="")  # To hold the selected checkbox value
        self.create_dialog()

    def create_dialog(self):
        self.top = tk.Toplevel(self.master)
        self.top.title("Input Two Integers and Select Value")
        self.top.geometry("500x400")  # Set a reasonable initial size
        self.top.minsize(400, 200)  # Set minimum size for scaling

        # Set a larger font
        large_font = ("Arial", 12)

        # Create labels and entry fields
        tk.Label(self.top, text="Enter Low Threshold:", font=large_font).pack(padx=10, pady=10, fill=tk.X)
        self.first_entry = tk.Entry(self.top, font=large_font)
        self.first_entry.pack(padx=10, pady=5, fill=tk.X)

        tk.Label(self.top, text="Enter High Threshold:", font=large_font).pack(padx=10, pady=10, fill=tk.X)
        self.second_entry = tk.Entry(self.top, font=large_font)
        self.second_entry.pack(padx=10, pady=5, fill=tk.X)

        # Create radio buttons for 12um, 16um, and 20um
        tk.Label(self.top, text="Select a value:", font=large_font).pack(padx=10, pady=10)
        tk.Radiobutton(self.top, text="12um", variable=self.selected_value, value=0, font=large_font).pack(anchor=tk.W, padx=10)
        tk.Radiobutton(self.top, text="16um", variable=self.selected_value, value=1, font=large_font).pack(anchor=tk.W, padx=10)
        tk.Radiobutton(self.top, text="20um", variable=self.selected_value, value=2, font=large_font).pack(anchor=tk.W, padx=10)

        # Create OK and Cancel buttons
        button_frame = tk.Frame(self.top)
        button_frame.pack(pady=20)
        
        ok_button = tk.Button(button_frame, text="OK", command=self.on_ok, font=large_font, width=8)
        ok_button.pack(side=tk.LEFT, padx=10)

        cancel_button = tk.Button(button_frame, text="Cancel", command=self.on_cancel, font=large_font, width=8)
        cancel_button.pack(side=tk.LEFT, padx=10)

    def on_ok(self):
        try:
            first_value = int(self.first_entry.get())
            second_value = int(self.second_entry.get())
            self.result = (first_value, second_value)

            # Get the selected radio button value
            selected_value = self.selected_value.get()
            print(f"Selected value: {selected_value}")  # Print selected value for demonstration

            self.top.destroy()
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter valid integers.")

    def on_cancel(self):
        self.result = None
        self.top.destroy()