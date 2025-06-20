# AI Meme Generator - GUI Module
# This file creates the user interface for our meme generator
# It handles everything the user sees and interacts with

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os, threading
from datetime import datetime
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.matcher import match_template
from utils.data_loader import load_and_prepare_dataset

class EditableCanvas(tk.Canvas):
    """
    This class creates a special canvas where users can edit memes
    It allows cropping, adding text, drawing rectangles, and moving elements around
    """
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        # Store the current image and its original version
        self.image = self.original_image = None
        # Keep track of all text and shapes added to the meme
        self.elements, self.mode, self.selected_element, self.start_pos = [], "select", None, None
        
        # Set up mouse event handlers - this is how we detect clicks and drags
        for e, h in [("<Button-1>", self.on_click), ("<B1-Motion>", self.on_drag), ("<ButtonRelease-1>", self.on_release), ("<Double-Button-1>", self.edit_text)]:
            self.bind(e, h)
        
    def set_image(self, path):
        """Load a new meme image and reset all edits"""
        # Open the image twice - one for editing, one to keep the original
        self.original_image, self.image, self.elements = Image.open(path), Image.open(path).copy(), []
        self.display()
    
    def display(self):
        """Show the current image on the canvas, scaled to fit nicely"""
        if not self.image: return
        
        # Get the available space on the canvas
        w, h = self.winfo_width() - 20, self.winfo_height() - 20
        if w <= 1: return self.after(100, self.display)  # Wait if canvas isn't ready yet
        
        # Calculate how much to scale the image so it fits
        img_w, img_h = self.image.size
        self.scale = min(w/img_w, h/img_h, 1.0)  # Don't make it bigger than original
        new_w, new_h = int(img_w * self.scale), int(img_h * self.scale)
        
        # Resize the image and convert it for tkinter
        self.photo = ImageTk.PhotoImage(self.image.resize((new_w, new_h), Image.Resampling.LANCZOS))
        
        # Clear the canvas and show the image centered
        self.delete("all")
        self.offset_x, self.offset_y = (self.winfo_width() - new_w) // 2, (self.winfo_height() - new_h) // 2
        self.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.photo)
        
        # Redraw all the text and shapes on top of the image
        for e in self.elements: self.draw_element(e)
    
    def draw_element(self, e):
        """Draw a text or rectangle element on the canvas"""
        # Convert image coordinates to screen coordinates
        x, y = self.offset_x + e['x'] * self.scale, self.offset_y + e['y'] * self.scale
        
        if e['type'] == 'text':
            # Draw text with the right size and color
            e['id'] = self.create_text(x, y, text=e['text'], font=('Arial', max(8, int(e['size'] * self.scale)), 'bold'), fill=e['color'], anchor=tk.NW)
        else:
            # Draw FILLED rectangle - completely filled with the chosen color
            e['id'] = self.create_rectangle(x, y, self.offset_x + e['x2'] * self.scale, self.offset_y + e['y2'] * self.scale, fill=e['color'], outline=e['color'], width=1)

    def screen_to_img(self, x, y): 
        """Convert screen coordinates to image coordinates"""
        return max(0, (x - self.offset_x) / self.scale), max(0, (y - self.offset_y) / self.scale)
    
    def on_click(self, event):
        """Handle mouse clicks - start dragging, select elements, or add new ones"""
        self.start_pos = (event.x, event.y)
        
        if self.mode == "select":
            # Check if user clicked on any existing element
            self.selected_element = None
            for e in self.elements:
                if 'id' in e and (bbox := self.bbox(e['id'])) and bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]:
                    self.selected_element = e
                    break
        elif self.mode == "text": 
            # Add new text where user clicked
            self.add_text(*self.screen_to_img(event.x, event.y))
    
    def on_drag(self, event):
        """Handle mouse dragging - move elements or draw crop/rectangle selection"""
        if not self.start_pos: return
        
        if self.mode in ["crop", "rect"]:
            # Show a temporary rectangle while dragging
            if hasattr(self, 'temp'): self.delete(self.temp)
            self.temp = self.create_rectangle(*self.start_pos, event.x, event.y, outline="red" if self.mode == "crop" else "black", dash=(5,5) if self.mode == "crop" else None, width=2)
        elif self.mode == "select" and self.selected_element:
            # Move the selected element
            dx, dy = event.x - self.start_pos[0], event.y - self.start_pos[1]
            self.move(self.selected_element['id'], dx, dy)
            
            # Update the element's actual coordinates so export works correctly
            dx_img, dy_img = dx / self.scale, dy / self.scale
            if self.selected_element['type'] == 'text':
                self.selected_element['x'] += dx_img
                self.selected_element['y'] += dy_img
            else:  # rectangle
                self.selected_element['x'] += dx_img
                self.selected_element['y'] += dy_img
                self.selected_element['x2'] += dx_img
                self.selected_element['y2'] += dy_img
            
            self.start_pos = (event.x, event.y)
    
    def on_release(self, event):
        """Handle mouse release - finalize crop or rectangle creation"""
        if hasattr(self, 'temp'):
            # Convert screen coordinates to image coordinates
            x1, y1, x2, y2 = *self.screen_to_img(*self.start_pos), *self.screen_to_img(event.x, event.y)
            
            if self.mode == "crop" and abs(x2-x1) > 10 and abs(y2-y1) > 10: 
                # Crop the image if the selection is big enough
                self.crop_image(min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))
            elif self.mode == "rect": 
                # Add a new rectangle
                self.add_rect(x1, y1, x2, y2)
            
            # Clean up the temporary rectangle
            self.delete(self.temp); delattr(self, 'temp')
        self.start_pos = None
    
    def add_text(self, x, y, text="New Text"):
        """Add a new text element and immediately open the editor"""
        e = {'type': 'text', 'x': x, 'y': y, 'text': text, 'size': 24, 'color': 'white'}
        self.elements.append(e); self.display(); self.edit_text_dialog(e)
    
    def add_rect(self, x1, y1, x2, y2):
        """Add a new filled rectangle with color choice"""
        def choose_color_and_add():
            # Create a small dialog to pick rectangle color
            dlg = tk.Toplevel(self)
            dlg.title("Select Rectangle Color")
            dlg.geometry("250x100")
            dlg.grab_set()
            dlg.resizable(False, False)

            tk.Label(dlg, text="Choose fill color:").pack(pady=10)
            btn_frame = tk.Frame(dlg)
            btn_frame.pack()

            def select(c):
                # Add the FILLED rectangle with the chosen color
                self.elements.append({'type': 'rect', 'x': min(x1,x2), 'y': min(y1,y2), 'x2': max(x1,x2), 'y2': max(y1,y2), 'color': c})
                self.display()
                dlg.destroy()

            tk.Button(btn_frame, text="Black", width=10, command=lambda: select("black")).pack(side=tk.LEFT, padx=10)
            tk.Button(btn_frame, text="White", width=10, command=lambda: select("white")).pack(side=tk.LEFT, padx=10)

        choose_color_and_add()
        self.display()
    
    def edit_text(self, event):
        """Handle double-clicks on text to edit it"""
        if self.mode == "select":
            # Find which text element was double-clicked
            for e in self.elements:
                if e['type'] == 'text' and 'id' in e and (bbox := self.bbox(e['id'])) and bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]:
                    self.edit_text_dialog(e); break
    
    def edit_text_dialog(self, e):
        """Open a dialog to edit text properties"""
        d = tk.Toplevel(self); d.title("Edit Text"); d.geometry("300x180"); d.grab_set()
        
        # Create input fields for text, size, and color
        tv, sv, cv = tk.StringVar(value=e['text']), tk.IntVar(value=e['size']), tk.StringVar(value=e['color'])
        
        for label, widget in [("Text:", tk.Entry(d, textvariable=tv, width=30)), ("Size:", tk.Spinbox(d, from_=8, to=72, textvariable=sv, width=10))]:
            tk.Label(d, text=label).pack(pady=2); widget.pack(pady=2)
            if label == "Text:": widget.focus_set()
        
        # Color selection radio buttons
        tk.Label(d, text="Color:").pack(pady=2)
        cf = tk.Frame(d); cf.pack(pady=2)
        for color in ["white", "black"]: tk.Radiobutton(cf, text=color.title(), variable=cv, value=color).pack(side=tk.LEFT, padx=10)
        
        # OK and Cancel buttons
        bf = tk.Frame(d); bf.pack(pady=10)
        def save(): e.update({'text': tv.get(), 'size': sv.get(), 'color': cv.get()}); self.display(); d.destroy()
        tk.Button(bf, text="OK", command=save).pack(side=tk.LEFT, padx=5)
        tk.Button(bf, text="Cancel", command=d.destroy).pack(side=tk.LEFT, padx=5)
        d.bind('<Return>', lambda x: save())  # Allow Enter key to save
    
    def crop_image(self, x1, y1, x2, y2):
        """Crop the image and adjust all element positions"""
        self.image = self.image.crop((int(x1), int(y1), int(x2), int(y2)))
        
        # Move all elements to account for the cropped area
        for e in self.elements:
            if e['type'] == 'text': 
                e['x'] -= x1; e['y'] -= y1
            else:  # rectangle
                e['x'] -= x1; e['y'] -= y1
                e['x2'] -= x1; e['y2'] -= y1
        self.display()
    
    def undo_crop(self): 
        """Restore the original image and clear all edits"""
        if self.original_image: self.image, self.elements = self.original_image.copy(), []; self.display()
    
    def delete_selected(self):
        """Remove the currently selected element"""
        if self.selected_element: self.elements.remove(self.selected_element); self.selected_element = None; self.display()
    
    def export_image(self, filename):
        """Save the final meme with all text and shapes to a file"""
        try:
            # Create a copy of the image to draw on
            img = self.image.copy(); draw = ImageDraw.Draw(img)
            
            # Draw all elements onto the image
            for e in self.elements:
                if e['type'] == 'text':
                    # Try to use Arial font, fall back to default if not available
                    try: font = ImageFont.truetype("arial.ttf", e['size'])
                    except: font = ImageFont.load_default()
                    draw.text((e['x'], e['y']), e['text'], font=font, fill=e['color'])
                else: 
                    # Draw FILLED rectangle - completely filled with color
                    draw.rectangle([e['x'], e['y'], e['x2'], e['y2']], fill=e['color'])
            
            # Save as JPEG or PNG depending on file extension
            if filename.lower().endswith(('.jpg', '.jpeg')):
                rgb = Image.new('RGB', img.size, (255, 255, 255)); rgb.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None); rgb.save(filename, 'JPEG', quality=95)
            else: img.save(filename, 'PNG')
            return True
        except Exception as e: 
            print(f"Export error: {e}")
            return False


class MemeGeneratorApp:
    """
    Main application class that creates the user interface
    This handles the AI meme generation and coordinates all the different parts
    """
    def __init__(self, root, dataset_path='data/labels.csv', images_folder='images'):
        # Store important paths and settings
        self.root, self.dataset_path, self.images_folder = root, dataset_path, images_folder
        root.title("AI Meme Generator - Enhanced"); root.geometry("1000x700")
        
        # Show loading screen while AI models load in background
        self.show_loading(); threading.Thread(target=self.load_data, daemon=True).start()
    
    def show_loading(self):
        """Display a loading screen with progress bar"""
        f = ttk.Frame(self.root, padding="20"); f.pack(fill=tk.BOTH, expand=True)
        ttk.Label(f, text="Loading AI Models...", font=("Helvetica", 16, "bold")).pack(pady=20)
        p = ttk.Progressbar(f, length=300, mode='indeterminate'); p.pack(pady=20); p.start()
        self.status_label = ttk.Label(f, text="Initializing..."); self.status_label.pack(pady=10)
        self.loading_frame = f
    
    def load_data(self):
        """Load the dataset and train AI models (runs in background)"""
        try:
            # This calls our AI system to process the meme dataset
            self.df, self.models = load_and_prepare_dataset(self.dataset_path, status_callback=lambda msg: self.status_label.config(text=msg))
            # Once done, show the main interface
            self.root.after(1000, self.create_ui)
        except Exception as e: messagebox.showerror("Error", f"Failed to load: {e}")
    
    def create_ui(self):
        """Create the main user interface after models are loaded"""
        self.loading_frame.destroy()
        
        # Main container for everything
        mf = ttk.Frame(self.root, padding="10"); mf.pack(fill=tk.BOTH, expand=True)
        ttk.Label(mf, text="AI Meme Generator", font=("Helvetica", 18, "bold")).pack(pady=5)
        cf = ttk.Frame(mf); cf.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left panel with controls
        lp = ttk.Frame(cf, width=280); lp.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10)); lp.pack_propagate(False)
        
        # Text input section for meme generation
        gf = ttk.LabelFrame(lp, text="Generate", padding="10"); gf.pack(fill=tk.X, pady=(0,10))
        self.input_text = tk.Text(gf, height=3, font=("Helvetica", 10)); self.input_text.pack(fill=tk.X, pady=(0,5))
        bf = ttk.Frame(gf); bf.pack(fill=tk.X)
        ttk.Button(bf, text="Generate", command=self.generate).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(bf, text="Clear", command=self.clear).pack(side=tk.LEFT)
        
        # Editing tools section
        tf = ttk.LabelFrame(lp, text="Tools", padding="10"); tf.pack(fill=tk.X, pady=(0,10))
        self.mode_var = tk.StringVar(value="select")
        # Radio buttons for different editing modes
        for text, mode in [("Select", "select"), ("Crop", "crop"), ("Text", "text"), ("Rectangle", "rect")]:
            ttk.Radiobutton(tf, text=text, variable=self.mode_var, value=mode, command=self.change_mode).pack(anchor=tk.W)
        # Quick action buttons
        for text, cmd in [("Top Text", lambda: self.quick_text("top")), ("Bottom Text", lambda: self.quick_text("bottom")), ("Undo Crop", lambda: self.canvas.undo_crop()), ("Delete Selected", lambda: self.canvas.delete_selected())]:
            ttk.Button(tf, text=text, command=cmd).pack(fill=tk.X, pady=1)
        
        # Export section
        ef = ttk.LabelFrame(lp, text="Export", padding="10"); ef.pack(fill=tk.X)
        self.export_btn = ttk.Button(ef, text="Export Meme", command=self.export, state=tk.DISABLED); self.export_btn.pack(fill=tk.X, pady=5)
        
        # Main editing canvas on the right
        canf = ttk.LabelFrame(cf, text="Editor", padding="5"); canf.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas = EditableCanvas(canf, bg="white"); self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self.canvas.display())
        
        # Information panel showing AI matching details
        inf = ttk.LabelFrame(mf, text="Match Information", padding="5"); inf.pack(fill=tk.X, pady=5)
        self.info_label = ttk.Label(inf, text="Enter text to see match details", font=("Helvetica", 9), wraplength=900); self.info_label.pack(fill=tk.X, pady=2)
        
        # Status bar at the bottom
        self.status_bar = ttk.Label(mf, text="Ready", relief=tk.SUNKEN, anchor=tk.W); self.status_bar.pack(fill=tk.X, pady=(10,0))
    
    def change_mode(self):
        """Switch between different editing modes"""
        self.canvas.mode = self.mode_var.get()
        self.status_bar.config(text=f"Mode: {self.canvas.mode.title()}")
    
    def generate(self):
        """Use AI to find the best meme for the user's text"""
        text = self.input_text.get("1.0", tk.END).strip()
        if not text: return messagebox.showwarning("Input Required", "Please enter text!")
        
        try:
            # Call our AI matching system
            best_match, info = match_template(text, self.df, self.models)
            image_path = os.path.join(self.images_folder, best_match['image_name'])
            
            if os.path.exists(image_path):
                # Load the chosen meme into the editor
                self.canvas.set_image(image_path); self.export_btn.config(state=tk.NORMAL)
                
                # Show detailed information about why this meme was chosen
                info_text = f"Template: {best_match['image_name']} | Score: {info['combined_score']:.3f} | TF-IDF: {info['tfidf_score']:.3f} | Semantic: {info['semantic_score']:.3f}"
                if info['classifier_score'] > 0: info_text += f" | RF: {info['classifier_score']:.3f}"
                self.info_label.config(text=info_text); self.status_bar.config(text=f"Generated: {best_match['image_name']}")
            else: messagebox.showerror("Error", "Image not found!")
        except Exception as e: messagebox.showerror("Error", f"Generation failed: {e}")
    
    def quick_text(self, pos):
        """Add standard top or bottom text to a meme"""
        if not self.canvas.image: return messagebox.showwarning("No Image", "Generate a meme first!")
        w, h = self.canvas.image.size
        # Position text at top or bottom center
        x, y = (w//2 - 50, 20) if pos == "top" else (w//2 - 50, h - 50)
        self.canvas.add_text(x, y, "TOP TEXT" if pos == "top" else "BOTTOM TEXT")
    
    def export(self):
        """Save the finished meme to a file"""
        if not self.canvas.image: return messagebox.showerror("Error", "No image to export!")
        
        # Ask user where to save the file
        filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")], initialfile=f"meme_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        if filename:
            if self.canvas.export_image(filename): messagebox.showinfo("Success", "Meme exported!")
            else: messagebox.showerror("Error", "Export failed!")
    
    def clear(self):
        """Reset everything to start over"""
        self.input_text.delete("1.0", tk.END); self.canvas.delete("all")
        self.canvas.image = self.canvas.elements = None; self.export_btn.config(state=tk.DISABLED)
        self.info_label.config(text="Enter text to see match details"); self.status_bar.config(text="Ready")