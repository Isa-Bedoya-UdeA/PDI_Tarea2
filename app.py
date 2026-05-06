import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image
import os
import threading

import predict
import train
import data_loader

ctk.set_appearance_mode("dark")

APP_BG = "#0C1225"
SECTION_BG = "#24293E"
CYAN_ACCENT = "#19CAE1"
CARD_OFF = "#0A0E1F"
PREVIEW_BG = "#080C1F"

app = ctk.CTk()
app.geometry("1000x700")
app.title("Task 2 - Image Processing")
app.configure(fg_color=APP_BG)

app.grid_columnconfigure(0, weight=3)
app.grid_columnconfigure(1, weight=1)
app.grid_rowconfigure(0, weight=1)

filter_var = ctk.StringVar(value="hog")
model_var = ctk.StringVar(value="nn")

filter_cards = {}
model_cards = {}

current_image_path = None

def update_card_styles():
    for val, frame in filter_cards.items():
        if filter_var.get() == val:
            frame.configure(border_color=CYAN_ACCENT, border_width=2)
        else:
            frame.configure(border_color=CARD_OFF, border_width=0)
            
    for val, frame in model_cards.items():
        if model_var.get() == val:
            frame.configure(border_color=CYAN_ACCENT, border_width=2)
        else:
            frame.configure(border_color=CARD_OFF, border_width=0)

def select_card(variable, value):
    variable.set(value)
    update_card_styles()

def load_image_action():
    global current_image_path
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if file_path:
        current_image_path = file_path
        # Show image in UI
        img = Image.open(file_path)
        img_ctk = ctk.CTkImage(light_image=img, dark_image=img, size=(300, 200))
        image_label_placeholder.configure(text="", image=img_ctk)
        result_label.configure(text="Result: ---")
        confidence_label.configure(text="0%")

def explore_dataset_action():
    def run_explore():
        btn_explore.configure(state="disabled", text="Explorando...")
        try:
            if not os.path.exists(data_loader.DATASET_PATH):
                data_loader.download_dataset_from_roboflow()
            data_loader.explore_full_dataset(data_loader.DATASET_PATH, visualize=True)
            messagebox.showinfo("Explorar Dataset", "Exploración completada. Revisa la consola y las ventanas emergentes.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al explorar: {e}")
        finally:
            btn_explore.configure(state="normal", text="Explorar Dataset")
            
    threading.Thread(target=run_explore, daemon=True).start()

def check_trained_models(descriptors):
    existing = []
    missing = []
    for desc in descriptors:
        model_nn = os.path.join(predict.MODELS_PATH, f'{desc}_nn.pkl')
        model_svm = os.path.join(predict.MODELS_PATH, f'{desc}_svm.pkl')
        scaler = os.path.join(predict.MODELS_PATH, f'{desc}_scaler.pkl')
        if os.path.exists(model_nn) and os.path.exists(model_svm) and os.path.exists(scaler):
            existing.append(desc)
        else:
            missing.append(desc)
    return existing, missing

def show_plots_and_metrics(all_metrics):
    modal = ctk.CTkToplevel(app)
    modal.title("Métricas de Entrenamiento")
    modal.geometry("550x450")
    
    text_box = ctk.CTkTextbox(modal, width=530, height=430, fg_color=SECTION_BG, text_color="white", font=ctk.CTkFont(size=13))
    text_box.pack(padx=10, pady=10)
    
    content = ""
    for desc, metrics in all_metrics.items():
        content += f"=== {desc.upper()} ===\n"
        content += f"  Red Neuronal:\n"
        content += f"    Accuracy: {metrics['nn']['accuracy']:.4f}\n"
        content += f"    Precision: {metrics['nn']['precision']:.4f}\n"
        content += f"    Recall: {metrics['nn']['recall']:.4f}\n"
        content += f"    F1-Score: {metrics['nn']['f1_score']:.4f}\n"
        content += f"  SVM:\n"
        content += f"    Accuracy: {metrics['svm']['accuracy']:.4f}\n"
        content += f"    Precision: {metrics['svm']['precision']:.4f}\n"
        content += f"    Recall: {metrics['svm']['recall']:.4f}\n"
        content += f"    F1-Score: {metrics['svm']['f1_score']:.4f}\n\n"
    
    text_box.insert("1.0", content)
    text_box.configure(state="disabled")
    
    import matplotlib.pyplot as plt
    plt.show()

def run_training_process(descriptors_to_train, auto_train=False, callback=None):
    def run_train():
        if not auto_train:
            btn_train.configure(state="disabled")
            btn_train_all.configure(state="disabled")
        
        progress_bar.pack(pady=10)
        progress_bar.start()
        
        all_metrics = {}
        try:
            for desc in descriptors_to_train:
                if len(descriptors_to_train) > 1:
                    result_label.configure(text=f"Entrenando {desc.upper()}...")
                
                result = train.train_complete_pipeline(descriptor=desc, show_plots=True, block_plots=False)
                all_metrics[desc] = result['metrics']
            
            if callback:
                app.after(0, callback)
                
            app.after(0, lambda: show_plots_and_metrics(all_metrics))
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al entrenar: {e}")
        finally:
            progress_bar.stop()
            progress_bar.pack_forget()
            if not auto_train:
                btn_train.configure(state="normal", text="Entrenar Seleccionado")
                btn_train_all.configure(state="normal", text="Entrenar Todos")
            result_label.configure(text="Result: ---")

    if auto_train:
        result_label.configure(text=f"Entrenando {descriptors_to_train[0].upper()}...")
        threading.Thread(target=run_train, daemon=True).start()
    else:
        result_label.configure(text="Iniciando entrenamiento...")
        threading.Thread(target=run_train, daemon=True).start()

def train_action(auto_train=False, callback=None):
    desc = filter_var.get()
    to_train = [desc]
    
    if not auto_train:
        existing, _ = check_trained_models([desc])
        if existing:
            respuesta = messagebox.askyesno("Modelos existentes", 
                f"El modelo para {desc.upper()} ya está entrenado.\n\n¿Deseas sobreescribirlo y entrenar nuevamente?")
            if not respuesta:
                if callback:
                    callback()
                return

    run_training_process(to_train, auto_train=auto_train, callback=callback)

def train_all_action():
    descriptors = ['hog', 'lbp', 'hog+lbp', 'lbp+hog']
    existing, missing = check_trained_models(descriptors)
    to_train = descriptors
    
    if existing:
        nombres = ", ".join([d.upper() for d in existing])
        respuesta = messagebox.askyesno("Modelos existentes", 
            f"Ya existen modelos entrenados para: {nombres}.\n\n¿Deseas sobreescribirlos todos?\n(Si eliges 'No', solo se entrenarán los faltantes)")
        if not respuesta:
            to_train = missing
            if not to_train:
                messagebox.showinfo("Aviso", "Todos los modelos ya están entrenados.")
                return

    run_training_process(to_train, auto_train=False, callback=None)

def classify_action():
    if not current_image_path:
        messagebox.showwarning("Aviso", "Por favor, carga una imagen primero.")
        return
        
    desc = filter_var.get()
    mod = model_var.get()
    
    # Check if model exists
    model_filename = f'{desc}_{mod}.pkl'
    scaler_filename = f'{desc}_scaler.pkl'
    
    model_path = os.path.join(predict.MODELS_PATH, model_filename)
    scaler_path = os.path.join(predict.MODELS_PATH, scaler_filename)
    
    def do_predict():
        res = predict.predict_image(current_image_path, descriptor=desc, model_type=mod)
        if res:
            result_label.configure(text=f"Result: {res['class_name']}")
            confidence_label.configure(text=f"{res['confidence']*100:.1f}%")
        else:
            result_label.configure(text="Result: Error")
            confidence_label.configure(text="0%")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        # Auto train
        train_action(auto_train=True, callback=do_predict)
    else:
        do_predict()

def clear_action():
    global current_image_path
    current_image_path = None
    filter_var.set("hog")
    model_var.set("nn")
    image_label_placeholder.configure(text="[ Image Preview ]", image="")
    result_label.configure(text="Result: ---")
    confidence_label.configure(text="0%")
    update_card_styles()

def create_radio_card(parent, text, variable, value, dict_ref):
    card = ctk.CTkFrame(parent, 
                        fg_color=CARD_OFF, 
                        border_color=CARD_OFF, 
                        height=50, 
                        corner_radius=8,
                        cursor="hand2")
    
    card.pack(fill="x", pady=4, padx=10)
    card.pack_propagate(False)
    
    lbl = ctk.CTkLabel(card, text=text, font=ctk.CTkFont(size=13), text_color="white")
    lbl.pack(side="left", padx=15)
    
    rb = ctk.CTkRadioButton(card, text="", variable=variable, value=value, 
                            width=20, border_color=CYAN_ACCENT, 
                            hover_color=CYAN_ACCENT, fg_color=CYAN_ACCENT,
                            command=update_card_styles)
    rb.pack(side="right", padx=15)

    card.bind("<Button-1>", lambda e: select_card(variable, value))
    lbl.bind("<Button-1>", lambda e: select_card(variable, value))
    
    dict_ref[value] = card
    return card

section_frame = ctk.CTkFrame(app, corner_radius=15, fg_color=SECTION_BG)
section_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

title = ctk.CTkLabel(section_frame, text="Coffee Bean Classifier", 
                    font=ctk.CTkFont(size=22, weight="bold"), text_color="white")
title.pack(pady=(20, 10))

image_display = ctk.CTkFrame(section_frame, width=400, height=250, fg_color=PREVIEW_BG)
image_display.pack(pady=10)
image_display.pack_propagate(False)
image_label_placeholder = ctk.CTkLabel(image_display, text="[ Image Preview ]", text_color="gray")
image_label_placeholder.place(relx=0.5, rely=0.5, anchor="center")

btn_load = ctk.CTkButton(section_frame, text="Load Image", 
                        command=load_image_action,
                        fg_color=CYAN_ACCENT, text_color="black", 
                        hover_color="#14A3B6", font=ctk.CTkFont(weight="bold"))
btn_load.pack(pady=15)

result_container = ctk.CTkFrame(section_frame, fg_color="transparent")
result_container.pack(pady=10)

result_label = ctk.CTkLabel(result_container, text="Result: ---", 
                                font=ctk.CTkFont(size=17), text_color="white")
result_label.pack()

confidence_label = ctk.CTkLabel(result_container, text="0%", 
                                font=ctk.CTkFont(size=28, weight="bold"), text_color=CYAN_ACCENT)
confidence_label.pack()

progress_bar = ctk.CTkProgressBar(section_frame, width=300, fg_color=CARD_OFF, progress_color=CYAN_ACCENT, mode="indeterminate")

# Dataset and Training Section
dataset_frame = ctk.CTkFrame(section_frame, fg_color="transparent")
dataset_frame.pack(side="bottom", pady=20, fill="x")

btn_explore = ctk.CTkButton(dataset_frame, text="Explorar Dataset", 
                        command=explore_dataset_action,
                        fg_color="#3A4A7A", text_color="white", 
                        hover_color="#2A3A6A", font=ctk.CTkFont(weight="bold"))
btn_explore.pack(side="left", padx=(20, 10), expand=True)

btn_train = ctk.CTkButton(dataset_frame, text="Entrenar Seleccionado", 
                        command=lambda: train_action(auto_train=False),
                        fg_color="#3A4A7A", text_color="white", 
                        hover_color="#2A3A6A", font=ctk.CTkFont(weight="bold"))
btn_train.pack(side="left", padx=10, expand=True)

btn_train_all = ctk.CTkButton(dataset_frame, text="Entrenar Todos", 
                        command=train_all_action,
                        fg_color="#3A4A7A", text_color="white", 
                        hover_color="#2A3A6A", font=ctk.CTkFont(weight="bold"))
btn_train_all.pack(side="left", padx=(10, 20), expand=True)


aside_frame = ctk.CTkFrame(app, corner_radius=0, fg_color=SECTION_BG)
aside_frame.grid(row=0, column=1, sticky="nsew")

def add_sidebar_title(parent, text):
    lbl = ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(size=16, weight="bold"), text_color="white")
    lbl.pack(pady=(25, 2), padx=15, anchor="w")
    line = ctk.CTkFrame(parent, height=2, fg_color=CYAN_ACCENT, width=50)
    line.pack(padx=15, anchor="w", pady=(0, 10))

add_sidebar_title(aside_frame, "Filters")
create_radio_card(aside_frame, "HOG", filter_var, "hog", filter_cards)
create_radio_card(aside_frame, "LBP", filter_var, "lbp", filter_cards)
create_radio_card(aside_frame, "HOG + LBP", filter_var, "hog+lbp", filter_cards)
create_radio_card(aside_frame, "LBP + HOG", filter_var, "lbp+hog", filter_cards)

add_sidebar_title(aside_frame, "Model")
create_radio_card(aside_frame, "Neural Network", model_var, "nn", model_cards)
create_radio_card(aside_frame, "SVM", model_var, "svm", model_cards)

btn_classify = ctk.CTkButton(aside_frame, text="Classify Image", 
                                command=classify_action, 
                                fg_color=CYAN_ACCENT, text_color="black",
                                hover_color="#14A3B6", font=ctk.CTkFont(weight="bold"))
btn_classify.pack(side="bottom", pady=(10, 30), padx=20, fill="x")

btn_clear = ctk.CTkButton(aside_frame, text="Clear", 
                            command=clear_action, 
                            fg_color="transparent", border_color=CYAN_ACCENT,
                            border_width=2, text_color=CYAN_ACCENT,
                            hover_color="#1a3a4a", font=ctk.CTkFont(weight="bold"))
btn_clear.pack(side="bottom", pady=10, padx=20, fill="x")

update_card_styles()

if __name__ == "__main__":
    app.mainloop()