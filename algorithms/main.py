import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk, Image
from ClassifyWithLinearSVC import linear_svc
from ClassifyWithMultinomialNB import naive_bayes

prediction = []
final_prediction = ''
selection = 0


# Function to display the result in a message box
def display_result():
    if selection != 0:
        if prediction.__len__() != 0:
            result = f"\n\n \t\t\tSorry, you have:  {prediction[0][0]}\t\t\t \n \t\t\t\t"
            messagebox.showinfo("Result", result)
        else:
            messagebox.showinfo("Result", "Please re-select checkup technique")
    else:
        messagebox.showinfo("Note", "Please select an algorithm")


# Function to handle radio button selection
def handle_radio_selection():

    if prediction.__len__() != 0:
        prediction.clear()

    global final_prediction, selection
    if radio_var.get() == 1:
        selection = 1
        if input_field1.get("1.0", tk.END).strip().__len__() == 0:
            messagebox.showinfo("Result", "Please specify what you feel before test")
        else:
            final_prediction = linear_svc(input_field1.get("1.0", tk.END).strip())
            prediction.append(final_prediction)

    elif radio_var.get() == 2:
        selection = 2
        if input_field1.get("1.0", tk.END).strip().__len__() == 0:
            messagebox.showinfo("Result", "Please specify what you feel before test")
        else:
            final_prediction = naive_bayes(input_field1.get("1.0", tk.END).strip())
            prediction.append(final_prediction)





# Create the main window
window = tk.Tk()
window.title("Symptom2Disease")

# Load and resize the image banner
image_path = '/home/mustafa7egazi/PycharmProjects/pythonProject/assetImages/symptoms.jpg'
image = Image.open(image_path)
resized_image = image.resize((900, 400))  # Set the desired size
banner_image = ImageTk.PhotoImage(resized_image)
banner_label = tk.Label(window, image=banner_image)
banner_label.pack()

# Create the text area
input_label1 = tk.Label(window, text="What do you feel?", font=("Arial", 20))
input_label1.pack(pady=10)
input_field1 = tk.Text(window, width=50, height=5,
                       font=("Arial", 16))  # Increase the width and set font size and height
input_field1.pack(pady=10)  # Add vertical margin

# Create a frame for the radio buttons
radio_frame = tk.Frame(window)
radio_frame.pack()

# Create the radio buttons
radio_var = tk.IntVar()
radio_button1 = tk.Radiobutton(radio_frame, font=("Arial", 16), text="LinearSVC", variable=radio_var,
                               value=1, command=handle_radio_selection)
radio_button1.pack(side=tk.LEFT)
radio_button2 = tk.Radiobutton(radio_frame, font=("Arial", 16), text="MultinomialNB", variable=radio_var,
                               value=2, command=handle_radio_selection)
radio_button2.pack(side=tk.LEFT)

# Create a margin between the radio buttons and the button
margin = tk.Label(window, text="")
margin.pack()

# Create the button
button = tk.Button(window, text="Show Result", command=display_result)
button.pack()

# Run the main event loop
window.mainloop()
