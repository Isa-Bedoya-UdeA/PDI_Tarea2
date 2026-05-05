import tkinter as tk
import customtkinter as ctk

ctk.set_appearance_mode("dark")

APP_BG = "#0C1225"
SECTION_BG = "#24293E"
CYAN_ACCENT = "#19CAE1"
CARD_OFF = "#0A0E1F"
PREVIEW_BG = "#080C1F"

app = ctk.CTk()
app.geometry("800x600")
app.title("Task 2 - Image Processing")
app.configure(fg_color=APP_BG)

app.grid_columnconfigure(0, weight=3)
app.grid_columnconfigure(1, weight=1)
app.grid_rowconfigure(0, weight=1)

filter_var = ctk.StringVar(value="HOG")
model_var = ctk.StringVar(value="Neural Network")

filter_cards = {}
model_cards = {}

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

def classify_action():
    result_text = "Coffee with borer"
    confidence_value = "85%"
    result_label.configure(text=f"Result: {result_text}")
    confidence_label.configure(text=confidence_value)

def clear_action():
    filter_var.set("HOG")
    model_var.set("Neural Network")
    result_label.configure(text="Result: ---")
    confidence_label.configure(text="0%")
    update_card_styles()

def create_radio_card(parent, text, variable, value, dict_ref):
    card = ctk.CTkFrame(parent, 
                        fg_color=CARD_OFF, 
                        border_color=CARD_OFF, 
                        height=60, 
                        corner_radius=8,
                        cursor="hand2")
    
    card.pack(fill="x", pady=5, padx=10)
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

title = ctk.CTkLabel(section_frame, text="Filters and Descriptors", 
                    font=ctk.CTkFont(size=22, weight="bold"), text_color="white")
title.pack(pady=(20, 10))

image_display = ctk.CTkFrame(section_frame, width=400, height=250, fg_color=PREVIEW_BG)
image_display.pack(pady=10)
image_label_placeholder = ctk.CTkLabel(image_display, text="[ Image Preview ]", text_color="gray")
image_label_placeholder.place(relx=0.5, rely=0.5, anchor="center")

btn_load = ctk.CTkButton(section_frame, text="Load Image", 
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

aside_frame = ctk.CTkFrame(app, corner_radius=0, fg_color=SECTION_BG)
aside_frame.grid(row=0, column=1, sticky="nsew")

def add_sidebar_title(parent, text):
    lbl = ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(size=16, weight="bold"), text_color="white")
    lbl.pack(pady=(25, 2), padx=15, anchor="w")
    line = ctk.CTkFrame(parent, height=2, fg_color=CYAN_ACCENT, width=50)
    line.pack(padx=15, anchor="w", pady=(0, 10))

add_sidebar_title(aside_frame, "Filters")
create_radio_card(aside_frame, "HOG", filter_var, "HOG", filter_cards)
create_radio_card(aside_frame, "LBP", filter_var, "LBP", filter_cards)

add_sidebar_title(aside_frame, "Model")
create_radio_card(aside_frame, "Neural Network", model_var, "Neural Network", model_cards)
create_radio_card(aside_frame, "SVM", model_var, "SVM", model_cards)

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

app.mainloop()