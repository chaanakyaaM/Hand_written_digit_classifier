import numpy as np
import keras as kr
import tkinter as tk
from PIL import Image

canvas = None
root = None
drawing = []
prediction_label = None

def paint(event):
    # Draws a small white circle where the mouse moves
    x1, y1 = (event.x - 6), (event.y - 6)
    x2, y2 = (event.x + 6), (event.y + 6)
    canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')
    drawing.append((event.x, event.y))

def predict_values():
    # To predict the drawn digit
    global prediction_label

    if canvas is None:
        print("Canvas not initialized.")
        return

    if not drawing:
        prediction_label.config(text="Please draw a digit first.")
        return

    try:
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        img_array = np.zeros((height, width), dtype=np.uint8)

        # Draw circles for the white strokes
        for x, y in drawing:
            for dy in range(-6, 7):
                for dx in range(-6, 7):
                    if dx * dx + dy * dy <= 36:
                        px, py = x + dx, y + dy
                        if 0 <= px < width and 0 <= py < height:
                            img_array[py, px] = 255

        # Crop the drawn digit
        coords = np.column_stack(np.where(img_array > 0))
        if coords.size > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(width - 1, x_max + padding)
            y_max = min(height - 1, y_max + padding)
            cropped_array = img_array[y_min:y_max + 1, x_min:x_max + 1]
        else:
            cropped_array = img_array

        # Make it square and resize to 28x28
        h, w = cropped_array.shape
        max_dim = max(h, w)
        square_array = np.zeros((max_dim, max_dim), dtype=np.uint8)
        y_offset = (max_dim - h) // 2
        x_offset = (max_dim - w) // 2
        square_array[y_offset:y_offset + h, x_offset:x_offset + w] = cropped_array

        square_image = Image.fromarray(square_array)
        resized_image = square_image.resize((28, 28), Image.LANCZOS)
        pixel_array = np.array(resized_image)
        pixel_array[pixel_array < 50] = 0

        # Convert array for model prediction
        val = np.expand_dims(pixel_array, axis=0)
        val = np.expand_dims(val, axis=-1)  

        # Load model and predict
        model = kr.models.load_model("CNN_225k_param_model.keras")
        prediction = model.predict(val)
        pred_number = np.argmax(prediction[0])

        # Show result below canvas
        prediction_label.config(
            text=f"Prediction: {pred_number}",
            fg="black",
            font=("Helvetica", 28, "bold")
        )
        print(f"Predicted number: {pred_number}")

    except Exception as e:
        print(f"Error: {e}")
        prediction_label.config(text=f"Error: {e}", fg="red", font=("Helvetica", 12))

def clear():
    """Clear the canvas and reset prediction label."""
    global drawing
    canvas.delete("all")
    drawing = []
    prediction_label.config(text="")

# Main
if __name__ == "__main__":
    root = tk.Tk()
    root.title("MNIST-style Digit Recognizer")

    canvas = tk.Canvas(root, bg='black', width=280, height=280)
    canvas.pack(pady=10)
    canvas.bind('<B1-Motion>', paint)

    # Buttons
    tk.Button(root, text="Predict", command=predict_values).pack(pady=5)
    tk.Button(root, text="Clear", command=clear).pack(pady=5)

    # Prediction label 
    prediction_label = tk.Label(root, text="", font=("Helvetica", 28, "bold"))
    prediction_label.pack(pady=10)

    root.mainloop()
