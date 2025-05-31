import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import random
import os
import json
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.feature import local_binary_pattern
from skimage.transform import resize

class FaceSketchApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Forensic Face Sketch App")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')

        # Initialize captcha variable
        self.correct_captcha = ""  

        # Setup data directories
        self.app_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = self.app_dir  
        os.makedirs(self.data_dir, exist_ok=True)

        # Paths
        self.credentials_file = os.path.join(self.data_dir, "saved_credentials.json")
        self.captcha_path = os.path.join(self.data_dir, "captcha.png")

        self.load_saved_credentials()
        self.setup_styles()
        self.create_login_frame()

    def setup_styles(self):
        style = ttk.Style()
        style.configure('Custom.TButton', padding=10, font=('Helvetica', 12))
        style.configure('Title.TLabel', font=('Helvetica', 24, 'bold'), padding=20, background='#f0f0f0')
        style.configure('Menu.TButton', padding=20, font=('Helvetica', 14))
        style.configure('Login.TFrame', background='#ffffff')

    def load_saved_credentials(self):
        try:
            if os.path.exists(self.credentials_file):
                with open(self.credentials_file, 'r') as f:
                    self.saved_credentials = json.load(f)
            else:
                self.saved_credentials = {}
        except:
            self.saved_credentials = {}

    def save_credentials(self, username, password):
        self.saved_credentials = {'username': username, 'password': password}
        with open(self.credentials_file, 'w') as f:
            json.dump(self.saved_credentials, f)

    def clear_saved_credentials(self):
        if os.path.exists(self.credentials_file):
            os.remove(self.credentials_file)
        self.saved_credentials = {}

    def create_login_frame(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        self.main_frame = ttk.Frame(self.root, style='Login.TFrame')
        self.main_frame.place(relx=0.5, rely=0.5, anchor="center")

        title = ttk.Label(self.main_frame, text="Forensic Face Sketch App", style='Title.TLabel')
        title.pack(pady=20)

        subtitle = ttk.Label(self.main_frame, text="Forensic Face Sketch System", font=('Helvetica', 14), background='#ffffff')
        subtitle.pack(pady=(0, 30))

        self.login_frame = ttk.Frame(self.main_frame, style='Login.TFrame')
        self.login_frame.pack(padx=40, pady=20)

        ttk.Label(self.login_frame, text="Username:", font=('Helvetica', 12), background='#ffffff').pack(anchor='w')
        self.username_entry = ttk.Entry(self.login_frame, width=30)
        self.username_entry.pack(fill=tk.X, pady=5)

        ttk.Label(self.login_frame, text="Password:", font=('Helvetica', 12), background='#ffffff').pack(anchor='w')
        self.password_entry = ttk.Entry(self.login_frame, width=30, show="*")
        self.password_entry.pack(fill=tk.X, pady=5)

        self.remember_var = tk.BooleanVar(value=bool(self.saved_credentials))
        ttk.Checkbutton(self.login_frame, text="Remember Me", variable=self.remember_var).pack(side=tk.LEFT)

        ttk.Button(self.login_frame, text="Login", style='Custom.TButton', command=self.verify_login).pack(pady=20, fill=tk.X)

        if self.saved_credentials:
            self.username_entry.insert(0, self.saved_credentials.get('username', ''))
            self.password_entry.insert(0, self.saved_credentials.get('password', ''))

    def verify_login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()

        if username and password:
            if self.remember_var.get():
                self.save_credentials(username, password)
            else:
                self.clear_saved_credentials()
            self.show_captcha()
        else:
            messagebox.showerror("Error", "Please enter both username and password")

    def generate_captcha(self):
        img = Image.new('RGB', (150, 50), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        captcha_text = str(random.randint(1000, 9999))
        self.correct_captcha = captcha_text  

        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except:
            font = None

        draw.text((40, 10), captcha_text, fill=(0, 0, 0), font=font)
        img.save(self.captcha_path)

    def show_captcha(self):
        self.generate_captcha()
        self.main_frame.place_forget()
        
        self.captcha_frame = ttk.Frame(self.root, style='Login.TFrame')
        self.captcha_frame.place(relx=0.5, rely=0.5, anchor="center")

        ttk.Label(self.captcha_frame, text="Captcha Verification", style='Title.TLabel').pack(pady=10)

        captcha_img = Image.open(self.captcha_path)
        captcha_img = ImageTk.PhotoImage(captcha_img)

        captcha_label = ttk.Label(self.captcha_frame, image=captcha_img)
        captcha_label.image = captcha_img
        captcha_label.pack(pady=10)

        ttk.Label(self.captcha_frame, text="Enter Captcha:", font=('Helvetica', 12), background='#ffffff').pack(pady=5)

        self.captcha_entry = ttk.Entry(self.captcha_frame, width=30, justify='center')
        self.captcha_entry.pack(pady=5)

        ttk.Button(self.captcha_frame, text="Verify Captcha", style='Custom.TButton', command=self.verify_captcha).pack(pady=20)
        ttk.Button(self.captcha_frame, text=" Back to Login", style='Custom.TButton', command=self.create_login_frame).pack(pady=10)

    def verify_captcha(self):
        if self.captcha_entry.get() == self.correct_captcha:
            self.captcha_frame.place_forget()
            self.create_main_menu()
        else:
            messagebox.showerror("Error", "Invalid Captcha")

    def create_main_menu(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        menu_frame = ttk.Frame(self.root)
        menu_frame.pack(pady=50)
    
        # Title
        title = ttk.Label(
            menu_frame,
            text="Forensic Face Sketch App",
            style='Title.TLabel'
        )
        title.pack(pady=20)
        
        # Upload Sketch button
        ttk.Button(
            menu_frame,
            text="Upload Sketch",
            style='Menu.TButton',
            command=self.show_main_application
        ).pack(pady=20)
        
        # Create New Sketch button
        ttk.Button(
            menu_frame,
            text="Create New Sketch",
            style='Menu.TButton',
            command=self.show_sketch_creation
        ).pack(pady=20)
        
        # Logout button
        logout_btn = ttk.Button(
            menu_frame,
            text="Logout",
            style='Menu.TButton',
            command=self.logout
        )
        logout_btn.pack(pady=10)
        
    def upload_sketch_window(self):
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Create main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Title
        title = ttk.Label(
            main_frame,
            text="Upload and Match Sketch",
            style='Title.TLabel'
        )
        title.pack(pady=20)
        
        # Instructions
        ttk.Label(
            main_frame,
            text="Upload a face sketch to find matching photos in the database.",
            font=('Helvetica', 12)
        ).pack(pady=10)
        
        # Upload frame
        upload_frame = ttk.Frame(main_frame)
        upload_frame.pack(pady=20)
        
        # Gender selection
        self.gender_var = tk.StringVar(value="all")
        gender_frame = ttk.Frame(upload_frame)
        gender_frame.pack(pady=10)
        
        ttk.Label(
            gender_frame,
            text="Filter by gender:",
            font=('Helvetica', 12)
        ).pack(side='left', padx=5)
        
        ttk.Radiobutton(
            gender_frame,
            text="All",
            variable=self.gender_var,
            value="all"
        ).pack(side='left', padx=5)
        
        ttk.Radiobutton(
            gender_frame,
            text="Male",
            variable=self.gender_var,
            value="male"
        ).pack(side='left', padx=5)
        
        ttk.Radiobutton(
            gender_frame,
            text="Female",
            variable=self.gender_var,
            value="female"
        ).pack(side='left', padx=5)
        
        # Upload section
        self.upload_label = ttk.Label(
            upload_frame,
            text="No sketch selected",
            font=('Helvetica', 12)
        )
        self.upload_label.pack(pady=10)
        
        # Preview frame
        self.preview_frame = ttk.Frame(upload_frame, borderwidth=1, relief='solid', width=300, height=300)
        self.preview_frame.pack_propagate(False)
        self.preview_frame.pack(pady=10)
        
        # Create empty preview label
        empty_img = Image.new('RGB', (300, 300), 'white')
        empty_photo = ImageTk.PhotoImage(empty_img)
        self.preview_label = ttk.Label(self.preview_frame, image=empty_photo)
        self.preview_label.image = empty_photo
        self.preview_label.pack(expand=True, fill='both')
        
        # Upload button
        upload_btn = ttk.Button(
            upload_frame,
            text="Select Sketch",
            style='Custom.TButton',
            command=self.upload_sketch
        )
        upload_btn.pack(pady=10)
        
        # Results frame with scrollbar
        results_container = ttk.Frame(main_frame)
        results_container.pack(pady=20, fill='both', expand=True)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(results_container)
        scrollbar.pack(side='right', fill='y')
        
        # Create canvas for scrolling
        self.results_canvas = tk.Canvas(results_container, yscrollcommand=scrollbar.set)
        self.results_canvas.pack(side='left', fill='both', expand=True)
        
        scrollbar.config(command=self.results_canvas.yview)
        
        # Create frame inside canvas for results
        self.results_frame = ttk.Frame(self.results_canvas)
        self.results_canvas.create_window((0, 0), window=self.results_frame, anchor='nw')
        
        # Configure canvas scrolling
        self.results_frame.bind('<Configure>', lambda e: self.results_canvas.configure(
            scrollregion=self.results_canvas.bbox('all')
        ))
        
        # Back button
        back_btn = ttk.Button(
            main_frame,
            text="Back to Menu",
            style='Custom.TButton',
            command=self.create_main_menu
        )
        back_btn.pack(pady=20)
        
    def upload_sketch(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if file_path:
            try:
                # Load and process the uploaded sketch
                uploaded_sketch = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if uploaded_sketch is None:
                    raise Exception("Failed to load image")
                
                # Update label with filename
                self.upload_label.config(text=f"Selected: {os.path.basename(file_path)}")
                
                # Ensure image is in uint8 format
                uploaded_sketch = cv2.normalize(uploaded_sketch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                # Resize to a standard size for comparison
                uploaded_sketch = cv2.resize(uploaded_sketch, (256, 256))
                
                # Clear previous results
                for widget in self.results_frame.winfo_children():
                    widget.destroy()
                
                # Compare with saved sketches
                self.compare_sketches(uploaded_sketch)
                
                # Display the uploaded sketch
                sketch_img = Image.open(file_path)
                sketch_img.thumbnail((300, 300))
                sketch_photo = ImageTk.PhotoImage(sketch_img)
                self.preview_label.configure(image=sketch_photo)
                self.preview_label.image = sketch_photo
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process sketch: {str(e)}")
    
    def compare_sketches(self, uploaded_sketch):
        results = []
        dataset_dir = os.path.join(self.data_dir, "dataset")
        
        # Get selected gender filter
        gender_filter = self.gender_var.get()
        
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
            
        if not os.path.exists(dataset_dir):
            ttk.Label(
                self.results_frame,
                text="Dataset not found. Please run setup_dataset.py first.",
                font=('Helvetica', 12)
            ).pack(pady=20)
            return
            
        # Extract features from uploaded sketch
        uploaded_features = self.extract_sketch_features(uploaded_sketch)
        
        # Compare with sketches in dataset
        genders = ['male', 'female'] if gender_filter == 'all' else [gender_filter]
        
        for gender in genders:
            sketches_dir = os.path.join(dataset_dir, gender, 'sketches')
            photos_dir = os.path.join(dataset_dir, gender, 'photos')
            
            if not os.path.exists(sketches_dir) or not os.path.exists(photos_dir):
                continue
                
            for sketch_file in os.listdir(sketches_dir):
                if not sketch_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                    
                # Load and process sketch
                sketch_path = os.path.join(sketches_dir, sketch_file)
                saved_sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
                if saved_sketch is None:
                    continue
                    
                saved_sketch = cv2.resize(saved_sketch, (256, 256))
                saved_features = self.extract_sketch_features(saved_sketch)
                
                # Calculate similarity
                score = self.calculate_similarity(uploaded_features, saved_features)
                
                # Find corresponding photo
                photo_file = sketch_file.replace('-sz1', '')  # Remove sketch suffix
                photo_path = os.path.join(photos_dir, photo_file)
                
                if os.path.exists(photo_path):
                    results.append({
                        'sketch_path': sketch_path,
                        'photo_path': photo_path,
                        'score': score,
                        'gender': gender
                    })
        
        # Sort results by similarity score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        if not results:
            ttk.Label(
                self.results_frame,
                text="No matches found. Try adjusting the gender filter.",
                font=('Helvetica', 12)
            ).pack(pady=20)
            return
            
        # Display results
        ttk.Label(
            self.results_frame,
            text=f"Found {len(results)} matches (showing top 10):",
            font=('Helvetica', 14, 'bold')
        ).pack(pady=10)
        
        # Create a frame for holding all result rows
        results_container = ttk.Frame(self.results_frame)
        results_container.pack(fill='both', expand=True)
        
        # Show top 10 matches
        for i, result in enumerate(results[:10], 1):
            try:
                result_frame = ttk.Frame(results_container)
                result_frame.pack(pady=10, padx=10, fill='x')
                
                # Load and resize sketch image
                sketch_img = Image.open(result['sketch_path'])
                sketch_img = sketch_img.convert('RGB')  # Convert to RGB mode
                sketch_img.thumbnail((100, 100))
                sketch_photo = ImageTk.PhotoImage(sketch_img)
                
                # Load and resize photo image
                photo_img = Image.open(result['photo_path'])
                photo_img = photo_img.convert('RGB')  # Convert to RGB mode
                photo_img.thumbnail((100, 100))
                photo_photo = ImageTk.PhotoImage(photo_img)
                
                # Keep references to prevent garbage collection
                result_frame.sketch_photo = sketch_photo
                result_frame.photo_photo = photo_photo
                
                # Create image labels
                sketch_label = ttk.Label(result_frame, image=sketch_photo)
                sketch_label.pack(side='left', padx=5)
                
                photo_label = ttk.Label(result_frame, image=photo_photo)
                photo_label.pack(side='left', padx=5)
                
                # Create info label
                info_frame = ttk.Frame(result_frame)
                info_frame.pack(side='left', padx=10, fill='x', expand=True)
                
                ttk.Label(
                    info_frame,
                    text=f"Match #{i} ({result['gender'].title()})",
                    font=('Helvetica', 12, 'bold')
                ).pack(anchor='w')
                
                ttk.Label(
                    info_frame,
                    text=f"Similarity: {result['score']:.1%}",
                    font=('Helvetica', 12)
                ).pack(anchor='w')
                
                # Add filename information
                ttk.Label(
                    info_frame,
                    text=f"File: {os.path.basename(result['photo_path'])}",
                    font=('Helvetica', 10)
                ).pack(anchor='w')
                
            except Exception as e:
                print(f"Error displaying result {i}: {str(e)}")
                continue
        
        # Update the canvas scroll region
        self.results_canvas.update_idletasks()
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox('all'))
        
    def extract_sketch_features(self, sketch):
        features = {}
        
        # Convert sketch to grayscale if it's not already
        if len(sketch.shape) == 3:
            sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
        
        # Resize to a fixed size
        sketch = cv2.resize(sketch, (128, 128))
        
        # Ensure the image is uint8 type
        if sketch.dtype != np.uint8:
            sketch = (sketch * 255).astype(np.uint8)
        
        # Calculate HOG features
        win_size = (128, 128)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        num_bins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
        features['hog'] = hog.compute(sketch).flatten()
        
        # Normalize HOG features
        norm = np.linalg.norm(features['hog'])
        if norm > 0:
            features['hog'] = features['hog'] / norm
        
        return features

    def calculate_similarity(self, features1, features2):
        """Calculate similarity between two sets of features."""
        try:
            # Calculate cosine similarity between HOG features
            similarity = np.dot(features1['hog'], features2['hog'])
            
            # Convert to percentage (0-1)
            similarity = max(0, min(similarity, 1))
            
            return similarity
            
        except Exception as e:
            print(f"Error in calculate_similarity: {str(e)}")
            return 0.0
        
    def compare_images(self, img1_path, img2_path):
        try:
            # Read images
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None or img2 is None:
                raise ValueError("Could not read one or both images")

            # Extract HOG features
            hog_features1 = self.extract_hog_features(img1)
            hog_features2 = self.extract_hog_features(img2)
            
            # Calculate cosine similarity between HOG features
            similarity = np.dot(hog_features1, hog_features2) / (
                np.linalg.norm(hog_features1) * np.linalg.norm(hog_features2)
            )
            
            # Calculate SSIM for additional comparison
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            img1_gray = cv2.resize(img1_gray, (128, 128))
            img2_gray = cv2.resize(img2_gray, (128, 128))
            ssim_score = ssim(img1_gray, img2_gray)
            
            # Combine both metrics
            combined_similarity = (similarity + ssim_score) / 2
            
            return combined_similarity
            
        except Exception as e:
            print(f"Error in compare_images: {str(e)}")
            return 0.0
        
    def show_main_application(self):
        # Clear any existing frames
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Create image frames container
        images_frame = ttk.Frame(main_frame)
        images_frame.pack(fill='x', pady=10)
        
        # Similarity label (initially hidden)
        self.similarity_label = ttk.Label(
            main_frame,
            text="",
            font=('Helvetica', 16, 'bold'),
            foreground='green'
        )
        self.similarity_label.pack(pady=5)
        
        # Create two image panels
        self.sketch_frame = ttk.Frame(images_frame, borderwidth=1, relief='solid', width=300, height=300)
        self.sketch_frame.pack_propagate(False)
        self.sketch_frame.pack(side='left', padx=10, pady=10)
        
        self.result_frame = ttk.Frame(images_frame, borderwidth=1, relief='solid', width=300, height=300)
        self.result_frame.pack_propagate(False)
        self.result_frame.pack(side='left', padx=10, pady=10)
        
        # Create empty labels for images (300x300 pixels)
        empty_img = Image.new('RGB', (300, 300), 'white')
        empty_photo = ImageTk.PhotoImage(empty_img)
        
        self.sketch_label = ttk.Label(self.sketch_frame, image=empty_photo)
        self.sketch_label.image = empty_photo
        self.sketch_label.pack(expand=True, fill='both')
        
        self.result_label = ttk.Label(self.result_frame, image=empty_photo)
        self.result_label.image = empty_photo
        self.result_label.pack(expand=True, fill='both')
        
        # Result info frame
        self.result_info_frame = ttk.Frame(main_frame)
        self.result_info_frame.pack(fill='x', pady=10)
        
        # Create buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(pady=20)
        
        # Add buttons with consistent size
        button_style = {'width': 15, 'style': 'Custom.TButton'}
        
        ttk.Button(
            buttons_frame,
            text="OPEN SKETCH",
            command=self.open_sketch,
            **button_style
        ).pack(side='left', padx=10)
        
        ttk.Button(
            buttons_frame,
            text="UPLOAD SKETCH",
            command=self.upload_sketch,
            **button_style
        ).pack(side='left', padx=10)
        
        ttk.Button(
            buttons_frame,
            text="FIND MATCH",
            command=self.find_match,
            **button_style
        ).pack(side='left', padx=10)
        
    def open_sketch(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if file_path:
            try:
                # Load and display the sketch
                sketch_img = Image.open(file_path)
                sketch_img.thumbnail((300, 300))
                sketch_photo = ImageTk.PhotoImage(sketch_img)
                self.sketch_label.configure(image=sketch_photo)
                self.sketch_label.image = sketch_photo
                
                # Update the current sketch
                self.current_sketch = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load sketch: {str(e)}")
                
    def upload_sketch(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if file_path:
            try:
                # Load and process the uploaded sketch
                uploaded_sketch = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if uploaded_sketch is None:
                    raise Exception("Failed to load image")
                
                # Ensure image is in uint8 format
                uploaded_sketch = cv2.normalize(uploaded_sketch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                # Resize to a standard size for comparison
                uploaded_sketch = cv2.resize(uploaded_sketch, (256, 256))
                
                # Update the current sketch
                self.current_sketch = uploaded_sketch
                
                # Display the uploaded sketch
                sketch_img = Image.open(file_path)
                sketch_img.thumbnail((300, 300))
                sketch_photo = ImageTk.PhotoImage(sketch_img)
                self.sketch_label.configure(image=sketch_photo)
                self.sketch_label.image = sketch_photo
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process sketch: {str(e)}")
                
    def find_match(self):
        if self.current_sketch is None:
            messagebox.showerror("Error", "Please load or upload a sketch first")
            return
        
        try:
            # Extract features from the current sketch
            sketch_features = self.extract_sketch_features(self.current_sketch)
            
            # Compare with saved sketches
            results = []
            dataset_dir = os.path.join(self.data_dir, "dataset")
            
            if not os.path.exists(dataset_dir):
                messagebox.showerror("Error", "Dataset not found. Please run setup_dataset.py first.")
                return
                
            # Compare with sketches in dataset
            for gender in ['male', 'female']:
                sketches_dir = os.path.join(dataset_dir, gender, 'sketches')
                photos_dir = os.path.join(dataset_dir, gender, 'photos')
                
                if not os.path.exists(sketches_dir) or not os.path.exists(photos_dir):
                    continue
                    
                for sketch_file in os.listdir(sketches_dir):
                    if not sketch_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        continue
                        
                    # Load and process sketch
                    sketch_path = os.path.join(sketches_dir, sketch_file)
                    saved_sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
                    if saved_sketch is None:
                        continue
                        
                    saved_sketch = cv2.resize(saved_sketch, (256, 256))
                    saved_features = self.extract_sketch_features(saved_sketch)
                    
                    # Calculate similarity
                    similarity = self.calculate_similarity(sketch_features, saved_features)
                    
                    # Find corresponding photo
                    photo_file = sketch_file.replace('-sz1', '')  # Remove sketch suffix
                    photo_path = os.path.join(photos_dir, photo_file)
                    
                    if os.path.exists(photo_path):
                        results.append({
                            'sketch_path': sketch_path,
                            'photo_path': photo_path,
                            'score': similarity * 100,  # Convert to percentage
                            'gender': gender
                        })
            
            # Sort results by similarity score
            results.sort(key=lambda x: x['score'], reverse=True)
            
            if not results:
                messagebox.showerror("Error", "No matches found")
                return
                
            # Display the top match
            top_match = results[0]
            self.similarity_label.config(text=f"Similarity: {top_match['score']:.1f}%")
            
            # Display the matched sketch and photo
            matched_sketch_img = Image.open(top_match['sketch_path'])
            matched_sketch_img.thumbnail((300, 300))
            matched_sketch_photo = ImageTk.PhotoImage(matched_sketch_img)
            self.sketch_label.configure(image=matched_sketch_photo)
            self.sketch_label.image = matched_sketch_photo
            
            matched_photo_img = Image.open(top_match['photo_path'])
            matched_photo_img.thumbnail((300, 300))
            matched_photo_photo = ImageTk.PhotoImage(matched_photo_img)
            self.result_label.configure(image=matched_photo_photo)
            self.result_label.image = matched_photo_photo
            
        except Exception as e:
            print(f"Error in find_match: {str(e)}")
            messagebox.showerror("Error", "Failed to process match")
            
    def logout(self):
        self.create_login_frame()
        
    def show_sketch_creation(self):
        # Clear any existing frames
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Create main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(expand=True, fill='both')
        
        # Initialize sketch canvas with the app reference
        self.sketch_canvas = FaceSketchCanvas(main_frame, self)
        
        # Add back button at the bottom
        back_btn = ttk.Button(
            main_frame,
            text="Back to Menu",
            style='Custom.TButton',
            command=self.create_main_menu
        )
        back_btn.pack(pady=10)
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = FaceSketchApp()
    app.run()
