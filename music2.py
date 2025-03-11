import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import pickle

class MusicPreferencesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Music Genre Predictor")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Variables
        self.df = None
        self.scaler = None
        self.model = None
        self.file_path = None
        
        # Create tabs
        self.tab_control = ttk.Notebook(self.root)
        
        self.tab1 = ttk.Frame(self.tab_control)
        self.tab2 = ttk.Frame(self.tab_control)
        self.tab3 = ttk.Frame(self.tab_control)
        self.tab4 = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.tab1, text='Data Analysis')
        self.tab_control.add(self.tab2, text='Model Training')
        self.tab_control.add(self.tab3, text='Prediction')
        self.tab_control.add(self.tab4, text='About')
        
        self.tab_control.pack(expand=1, fill="both")
        
        # Tab 1: Data Analysis
        self.setup_data_analysis_tab()
        
        # Tab 2: Model Training
        self.setup_model_training_tab()
        
        # Tab 3: Prediction
        self.setup_prediction_tab()
        
        # Tab 4: About
        self.setup_about_tab()
    
    def setup_data_analysis_tab(self):
        # Frame for controls
        control_frame = ttk.LabelFrame(self.tab1, text="Data Controls")
        control_frame.pack(fill="x", padx=10, pady=10)
        
        # Load data button
        ttk.Button(control_frame, text="Load Data", command=self.load_data).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Default data button
        ttk.Button(control_frame, text="Use Default Data", command=self.use_default_data).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Data info frame
        self.data_info_frame = ttk.LabelFrame(self.tab1, text="Data Information")
        self.data_info_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add a text widget to display data information
        self.data_info_text = tk.Text(self.data_info_frame, height=10)
        self.data_info_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Visualization frame
        self.viz_frame = ttk.LabelFrame(self.tab1, text="Visualizations")
        self.viz_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Initially show a message to load data
        ttk.Label(self.viz_frame, text="Load data to see visualizations").pack(pady=20)
    
    def setup_model_training_tab(self):
        # Frame for model options
        model_frame = ttk.LabelFrame(self.tab2, text="Model Options")
        model_frame.pack(fill="x", padx=10, pady=10)
        
        # Model selection
        ttk.Label(model_frame, text="Select Model:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.model_var = tk.StringVar(value="Random Forest")
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, state="readonly")
        model_combo['values'] = ('Gaussian Naive Bayes', 'K-Nearest Neighbors', 'Random Forest')
        model_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Test size slider
        ttk.Label(model_frame, text="Test Size:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        
        self.test_size_var = tk.DoubleVar(value=0.3)
        test_size_slider = ttk.Scale(model_frame, from_=0.1, to=0.5, length=200, 
                                     orient="horizontal", variable=self.test_size_var, 
                                     command=lambda s: self.test_size_var.set(round(float(s), 2)))
        test_size_slider.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        test_size_label = ttk.Label(model_frame, textvariable=tk.StringVar(value=f"Value: {self.test_size_var.get()}"))
        self.test_size_var.trace_add("write", lambda *args: test_size_label.config(text=f"Value: {self.test_size_var.get():.2f}"))
        test_size_label.grid(row=1, column=2, padx=5, pady=5, sticky="w")
        
        # Train button
        ttk.Button(model_frame, text="Train Model", command=self.train_model).grid(row=2, column=0, columnspan=3, padx=5, pady=10)
        
        # Results frame
        self.results_frame = ttk.LabelFrame(self.tab2, text="Training Results")
        self.results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Text widget for results
        self.results_text = tk.Text(self.results_frame, height=15)
        self.results_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Model operations frame
        model_ops_frame = ttk.Frame(self.tab2)
        model_ops_frame.pack(fill="x", padx=10, pady=10)
        
        # Save model button
        ttk.Button(model_ops_frame, text="Save Trained Model", command=self.save_model).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Load model button
        ttk.Button(model_ops_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=5, pady=5)
    
    def setup_prediction_tab(self):
        # Frame for input
        input_frame = ttk.LabelFrame(self.tab3, text="Make Predictions")
        input_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Age input
        ttk.Label(input_frame, text="Age:").grid(row=0, column=0, padx=5, pady=10, sticky="w")
        self.age_var = tk.IntVar(value=25)
        ttk.Spinbox(input_frame, from_=10, to=100, textvariable=self.age_var, width=5).grid(row=0, column=1, padx=5, pady=10, sticky="w")
        
        # Gender input
        ttk.Label(input_frame, text="Gender:").grid(row=1, column=0, padx=5, pady=10, sticky="w")
        self.gender_var = tk.IntVar(value=1)
        gender_frame = ttk.Frame(input_frame)
        gender_frame.grid(row=1, column=1, padx=5, pady=10, sticky="w")
        ttk.Radiobutton(gender_frame, text="Male", variable=self.gender_var, value=1).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(gender_frame, text="Female", variable=self.gender_var, value=0).pack(side=tk.LEFT, padx=5)
        
        # Predict button
        ttk.Button(input_frame, text="Predict Genre", command=self.predict_genre).grid(row=2, column=0, columnspan=2, padx=5, pady=10)
        
        # Results label
        self.prediction_result = ttk.Label(input_frame, text="", font=("Arial", 12, "bold"))
        self.prediction_result.grid(row=3, column=0, columnspan=2, padx=5, pady=10)
        
        # Information about other ages/genders
        self.prediction_info_frame = ttk.LabelFrame(self.tab3, text="Common Patterns")
        self.prediction_info_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.prediction_info = tk.Text(self.prediction_info_frame, height=8)
        self.prediction_info.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Initially display suggestion to train a model
        self.prediction_info.insert(tk.END, "Train a model in the 'Model Training' tab before making predictions.")
        self.prediction_info.config(state=tk.DISABLED)
    
    def setup_about_tab(self):
        # About information
        about_frame = ttk.Frame(self.tab4)
        about_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        about_text = """
        Pro tip: Maximize the program window

        Music Genre Preference Predictor
        ===============================
        
        This application demonstrates a complete machine learning workflow for predicting 
        music genre preferences based on age and gender.
        
        Features:
        - Data exploration and visualization
        - Model training with multiple algorithms
        - Interactive prediction system
        - Save/load trained models

        Training:
        - Gaussian Naive Bayes: Suitable for small datasets and categorical data. 
        - K-Nearest Neighbors: Works well for classification where similar examples are expected to have similar results. 
        - Random Forest: Robust against outliers and over-adaptation, good for many problems.
        
        Usage:
        1. Load your dataset (CSV file with age, gender, genre columns)
        2. Explore the data in the first tab
        3. Train a model in the second tab
        4. Make predictions in the third tab
        5. Save your model for future use or load a previously saved model
        
        Built with Python, scikit-learn, and Tkinter.
        """
        
        ttk.Label(about_frame, text=about_text, justify=tk.LEFT).pack(fill="both", expand=True)
    
    def load_data(self):
        """Load data from a CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.file_path = file_path
                self.display_data_info()
                self.create_visualizations()
                self.check_and_handle_data_issues()
                messagebox.showinfo("Success", "Data loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def use_default_data(self):
        """Create and use a default music.csv dataset"""
        # Create a default dataset
        default_data = {
            'age': [20, 23, 25, 26, 29, 30, 31, 33, 37, 20, 21, 25, 26, 27, 30, 31, 34, 35],
            'gender': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'genre': ['HipHop', 'HipHop', 'HipHop', 'Jazz', 'Jazz', 'Jazz', 'Classical', 'Classical', 
                      'Classical', 'Dance', 'Dance', 'Dance', 'Acoustic', 'Acoustic', 'Acoustic', 
                      'Classical', 'Classical', 'Classical']
        }
        
        self.df = pd.DataFrame(default_data)
        self.file_path = "Default Dataset"
        self.display_data_info()
        self.create_visualizations()
        self.check_and_handle_data_issues()
        messagebox.showinfo("Success", "Default data loaded!")
    
    def display_data_info(self):
        """Display dataset information in the text widget"""
        if self.df is not None:
            self.data_info_text.config(state=tk.NORMAL)
            self.data_info_text.delete(1.0, tk.END)
            
            # Basic info
            self.data_info_text.insert(tk.END, f"Dataset: {os.path.basename(self.file_path) if self.file_path != 'Default Dataset' else 'Default Dataset'}\n")
            self.data_info_text.insert(tk.END, f"Number of records: {len(self.df)}\n\n")
            
            # Summary statistics
            self.data_info_text.insert(tk.END, "Summary Statistics for Age:\n")
            age_stats = self.df['age'].describe()
            self.data_info_text.insert(tk.END, f"  Min: {age_stats['min']}\n")
            self.data_info_text.insert(tk.END, f"  Max: {age_stats['max']}\n")
            self.data_info_text.insert(tk.END, f"  Mean: {age_stats['mean']:.2f}\n")
            self.data_info_text.insert(tk.END, f"  Std: {age_stats['std']:.2f}\n\n")
            
            # Gender distribution
            gender_counts = self.df['gender'].value_counts()
            self.data_info_text.insert(tk.END, "Gender Distribution:\n")
            self.data_info_text.insert(tk.END, f"  Male (1): {gender_counts.get(1, 0)}\n")
            self.data_info_text.insert(tk.END, f"  Female (0): {gender_counts.get(0, 0)}\n\n")
            
            # Genre distribution
            genre_counts = self.df['genre'].value_counts()
            self.data_info_text.insert(tk.END, "Genre Distribution:\n")
            for genre, count in genre_counts.items():
                self.data_info_text.insert(tk.END, f"  {genre}: {count}\n")
            
            self.data_info_text.config(state=tk.DISABLED)

    def check_and_handle_data_issues(self):
        """Check and handle missing values and outliers in the dataset"""
        if self.df is None:
            return
    
        # Check for missing values
        missing_values = self.df.isnull().sum()
        has_missing = missing_values.sum() > 0
    
        # Check for outliers in age using IQR method
        Q1 = self.df['age'].quantile(0.25)
        Q3 = self.df['age'].quantile(0.75)
        IQR = Q3 - Q1
    
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
    
        age_outliers = self.df[(self.df['age'] < lower_bound) | (self.df['age'] > upper_bound)]
        has_outliers = len(age_outliers) > 0
    
        # No issues found
        if not has_missing and not has_outliers:
            messagebox.showinfo("Data Check", "No missing values or outliers found in the dataset.")
            return
    
        # Create a data issues report
        report = "Data Issues Report:\n\n"
    
        if has_missing:
            report += "Missing Values Found:\n"
            for col, count in missing_values.items():
                if count > 0:
                    report += f"  {col}: {count} missing values\n"
            report += "\n"
    
        if has_outliers:
            report += f"Age Outliers Found: {len(age_outliers)} records\n"
            report += f"  Age Range: {self.df['age'].min()} - {self.df['age'].max()}\n"
            report += f"  Expected Range: {lower_bound:.2f} - {upper_bound:.2f}\n\n"
    
        # Show dialog with the report and options
        self.show_data_issues_dialog(report, has_missing, has_outliers, Q1, Q3, IQR)

    def show_data_issues_dialog(self, report, has_missing, has_outliers, Q1, Q3, IQR):
        """Show dialog with data issues and handling options"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Data Issues Found")
        dialog.geometry("500x500")
        dialog.transient(self.root)
        dialog.grab_set()
    
        # Report text
        report_frame = ttk.LabelFrame(dialog, text="Issues Report")
        report_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        report_text = tk.Text(report_frame, height=10)
        report_text.pack(fill="both", expand=True, padx=5, pady=5)
        report_text.insert(tk.END, report)
        report_text.config(state=tk.DISABLED)
        
        # Options frame
        options_frame = ttk.LabelFrame(dialog, text="Handling Options")
        options_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
        # Missing values handling
        if has_missing:
            ttk.Label(options_frame, text="Handle Missing Values:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
            missing_var = tk.StringVar(value="mean")
            missing_combo = ttk.Combobox(options_frame, textvariable=missing_var, state="readonly")
            missing_combo['values'] = ('mean', 'median', 'mode', 'drop')
            missing_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
    
        # Outliers handling
        if has_outliers:
            ttk.Label(options_frame, text="Handle Age Outliers:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
            
            outlier_var = tk.StringVar(value="cap")
            outlier_combo = ttk.Combobox(options_frame, textvariable=outlier_var, state="readonly")
            outlier_combo['values'] = ('cap', 'mean', 'median', 'drop')
            outlier_combo.grid(row=1, column=1, padx=5, pady=5, sticky="w")
    
        # Buttons
        def apply_fixes():
            # Handle missing values
            if has_missing:
                method = missing_var.get()
                if method == 'mean':
                    self.df = self.df.fillna(self.df.mean(numeric_only=True))
                elif method == 'median':
                    self.df = self.df.fillna(self.df.median(numeric_only=True))
                elif method == 'mode':
                    self.df = self.df.fillna(self.df.mode().iloc[0])
                elif method == 'drop':
                    self.df = self.df.dropna()
        
            # Handle outliers
            if has_outliers:
                method = outlier_var.get()
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                if method == 'cap':
                    self.df.loc[self.df['age'] < lower_bound, 'age'] = lower_bound
                    self.df.loc[self.df['age'] > upper_bound, 'age'] = upper_bound
                elif method == 'mean':
                    mean_age = self.df['age'].mean()
                    self.df.loc[self.df['age'] < lower_bound, 'age'] = mean_age
                    self.df.loc[self.df['age'] > upper_bound, 'age'] = mean_age
                elif method == 'median':
                    median_age = self.df['age'].median()
                    self.df.loc[self.df['age'] < lower_bound, 'age'] = median_age
                    self.df.loc[self.df['age'] > upper_bound, 'age'] = median_age
                elif method == 'drop':
                    self.df = self.df[(self.df['age'] >= lower_bound) & (self.df['age'] <= upper_bound)]
        
            # Update displays
            self.display_data_info()
            self.create_visualizations()
            messagebox.showinfo("Success", "Data issues have been fixed!")
            dialog.destroy()
    
        def cancel():
            dialog.destroy()
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(btn_frame, text="Apply Fixes", command=apply_fixes).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=cancel).pack(side=tk.RIGHT, padx=5)
    
    def create_visualizations(self):
        """Create and display visualizations"""
        if self.df is not None:
            # Clear previous visualizations
            for widget in self.viz_frame.winfo_children():
                widget.destroy()
            
            # Create figure with subplots
            fig = Figure(figsize=(8, 8))
            
            # Age distribution
            ax1 = fig.add_subplot(221)
            ax1.hist(self.df['age'], bins=10, edgecolor='black')
            ax1.set_title('Age Distribution')
            ax1.set_xlabel('Age')
            ax1.set_ylabel('Count')
            
            # Genre distribution
            ax2 = fig.add_subplot(222)
            genre_counts = self.df['genre'].value_counts()
            ax2.bar(genre_counts.index, genre_counts.values)
            ax2.set_title('Genre Distribution')
            ax2.set_xlabel('Genre')
            ax2.set_ylabel('Count')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            
            # Age by genre boxplot
            ax3 = fig.add_subplot(223)
            genre_order = genre_counts.index.tolist()
            sns.boxplot(x='genre', y='age', data=self.df, ax=ax3, order=genre_order)
            ax3.set_title('Age Distribution by Genre')
            ax3.set_xlabel('Genre')
            ax3.set_ylabel('Age')
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
            
            # Genre by gender
            ax4 = fig.add_subplot(224)
            gender_genre = pd.crosstab(self.df['gender'], self.df['genre'])
            gender_genre.plot(kind='bar', stacked=True, ax=ax4)
            ax4.set_title('Genre Preferences by Gender')
            ax4.set_xlabel('Gender (0=Female, 1=Male)')
            ax4.set_ylabel('Count')
            ax4.legend(title='Genre')
            
            fig.tight_layout()
            
            # Add the plot to the GUI
            canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def train_model(self):
        """Train the selected machine learning model"""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
        
        try:
            # Clear previous results
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Training model...\n\n")
            self.results_text.update()
            
            # Prepare data
            X = self.df[['age', 'gender']]
            y = self.df['genre']
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            test_size = self.test_size_var.get()
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42)
            
            # Select model
            model_name = self.model_var.get()
            if model_name == 'Gaussian Naive Bayes':
                self.model = GaussianNB()
            elif model_name == 'K-Nearest Neighbors':
                self.model = KNeighborsClassifier(n_neighbors=3)
            else:  # Random Forest
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Evaluate
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            # Display results
            self.results_text.insert(tk.END, f"Model: {model_name}\n")
            self.results_text.insert(tk.END, f"Test Size: {test_size:.2f}\n\n")
            self.results_text.insert(tk.END, f"Accuracy: {acc:.4f}\n\n")
            self.results_text.insert(tk.END, "Classification Report:\n")
            self.results_text.insert(tk.END, f"{report}\n")
            
            self.results_text.insert(tk.END, "Confusion Matrix:\n")
            self.results_text.insert(tk.END, f"{cm}\n\n")
            
            # Feature importance for Random Forest
            if isinstance(self.model, RandomForestClassifier):
                self.results_text.insert(tk.END, "Feature Importance:\n")
                importance = self.model.feature_importances_
                for i, feat in enumerate(['Age', 'Gender']):
                    self.results_text.insert(tk.END, f"  {feat}: {importance[i]:.4f}\n")
            
            self.results_text.config(state=tk.DISABLED)
            
            # Update prediction info
            self.update_prediction_info()
            
            messagebox.showinfo("Success", f"{model_name} trained successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during training: {str(e)}")
    
    def update_prediction_info(self):
        """Update the prediction information text"""
        if self.model is None:
            return
            
        self.prediction_info.config(state=tk.NORMAL)
        self.prediction_info.delete(1.0, tk.END)
        
        self.prediction_info.insert(tk.END, "Common patterns found in the dataset:\n\n")
        
        # Get average age for each genre
        genre_ages = {}
        for genre in self.df['genre'].unique():
            avg_age = self.df[self.df['genre'] == genre]['age'].mean()
            genre_ages[genre] = avg_age
        
        # Sort genres by average age
        sorted_genres = sorted(genre_ages.items(), key=lambda x: x[1])
        
        # Display genres by typical age group
        self.prediction_info.insert(tk.END, "Genres by typical age group:\n")
        for genre, age in sorted_genres:
            self.prediction_info.insert(tk.END, f"  {genre}: average age {age:.1f}\n")
        
        # Get gender preferences
        self.prediction_info.insert(tk.END, "\nGender preferences:\n")
        
        # For males
        male_genres = self.df[self.df['gender'] == 1]['genre'].value_counts()
        top_male = male_genres.index[0] if len(male_genres) > 0 else "N/A"
        self.prediction_info.insert(tk.END, f"  Males tend to prefer: {top_male}\n")
        
        # For females
        female_genres = self.df[self.df['gender'] == 0]['genre'].value_counts()
        top_female = female_genres.index[0] if len(female_genres) > 0 else "N/A"
        self.prediction_info.insert(tk.END, f"  Females tend to prefer: {top_female}\n")
        
        self.prediction_info.config(state=tk.DISABLED)
    
    def predict_genre(self):
        """Make a prediction based on user input"""
        if self.model is None or self.scaler is None:
            messagebox.showwarning("Warning", "Please train a model first!")
            return
        
        try:
            # Get input values
            age = self.age_var.get()
            gender = self.gender_var.get()
            
            # Scale the input
            features = self.scaler.transform([[age, gender]])
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            
            # Get probabilities if available
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(features)[0]
                max_proba = max(proba) * 100
                confidence_text = f" (Confidence: {max_proba:.1f}%)"
            else:
                confidence_text = ""
            
            # Update result label
            gender_text = "Male" if gender == 1 else "Female"
            result_text = f"Predicted Genre: {prediction}{confidence_text}"
            self.prediction_result.config(text=result_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during prediction: {str(e)}")
    
    def save_model(self):
        """Save the trained model to a file"""
        if self.model is None:
            messagebox.showwarning("Warning", "Please train a model first!")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                title="Save Model",
                defaultextension=".pkl",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            
            if file_path:
                with open(file_path, 'wb') as f:
                    # Save both the model and the scaler
                    pickle.dump((self.model, self.scaler), f)
                messagebox.showinfo("Success", "Model saved successfully!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error saving model: {str(e)}")
    
    def load_model(self):
        """Load a trained model from a pickle file"""
        try:
            file_path = filedialog.askopenfilename(
                title="Load Model",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            
            if file_path:
                with open(file_path, 'rb') as f:
                    # Load both the model and the scaler
                    self.model, self.scaler = pickle.load(f)
                
                # Update the results text to show model information
                self.results_text.config(state=tk.NORMAL)
                self.results_text.delete(1.0, tk.END)
                
                # Display model information
                model_type = type(self.model).__name__
                self.results_text.insert(tk.END, f"Loaded Model: {model_type}\n\n")
                
                # Show model parameters if available
                self.results_text.insert(tk.END, "Model Parameters:\n")
                if hasattr(self.model, "get_params"):
                    params = self.model.get_params()
                    for param, value in params.items():
                        self.results_text.insert(tk.END, f"  {param}: {value}\n")
                
                # For Random Forest, show feature importance
                if isinstance(self.model, RandomForestClassifier) and hasattr(self.model, "feature_importances_"):
                    self.results_text.insert(tk.END, "\nFeature Importance:\n")
                    importance = self.model.feature_importances_
                    for i, feat in enumerate(['Age', 'Gender']):
                        self.results_text.insert(tk.END, f"  {feat}: {importance[i]:.4f}\n")
                
                # For Naive Bayes, show class priors
                if isinstance(self.model, GaussianNB) and hasattr(self.model, "class_prior_"):
                    self.results_text.insert(tk.END, "\nClass Priors:\n")
                    for i, cls in enumerate(self.model.classes_):
                        self.results_text.insert(tk.END, f"  {cls}: {self.model.class_prior_[i]:.4f}\n")
                
                self.results_text.insert(tk.END, f"\nModel loaded from: {os.path.basename(file_path)}")
                self.results_text.config(state=tk.DISABLED)
                
                # Provide information about what data the model might expect
                if self.df is None:
                    self.prediction_info.config(state=tk.NORMAL)
                    self.prediction_info.delete(1.0, tk.END)
                    self.prediction_info.insert(tk.END, "Model loaded successfully. Load a dataset to see data patterns.\n\n")
                    self.prediction_info.insert(tk.END, "You can now make predictions in the 'Prediction' tab.")
                    self.prediction_info.config(state=tk.DISABLED)
                else:
                    # Update prediction info with current dataset
                    self.update_prediction_info()
                
                messagebox.showinfo("Success", f"Model loaded successfully from {os.path.basename(file_path)}!")
                
                # If we're on the Model Training tab, switch to the Prediction tab
                if self.tab_control.index(self.tab_control.select()) == 1:  # Model Training tab
                    self.tab_control.select(2)  # Switch to Prediction tab
                
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {str(e)}\n\nThe file may be corrupted or incompatible with this application.")
            
    def model_info_dialog(self):
        """Show detailed information about the loaded model"""
        if self.model is None:
            messagebox.showwarning("Warning", "No model is loaded!")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Model Information")
        dialog.geometry("600x400")
        dialog.transient(self.root)
        
        # Text widget for model info
        info_text = tk.Text(dialog, wrap=tk.WORD, padx=10, pady=10)
        info_text.pack(fill="both", expand=True)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(info_text, command=info_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        info_text.config(yscrollcommand=scrollbar.set)
        
        # Display model information
        model_type = type(self.model).__name__
        info_text.insert(tk.END, f"Model Type: {model_type}\n\n")
        
        # Display classes
        if hasattr(self.model, "classes_"):
            info_text.insert(tk.END, "Classes:\n")
            for cls in self.model.classes_:
                info_text.insert(tk.END, f"  {cls}\n")
            info_text.insert(tk.END, "\n")
        
        # Show model parameters
        info_text.insert(tk.END, "Model Parameters:\n")
        if hasattr(self.model, "get_params"):
            params = self.model.get_params()
            for param, value in params.items():
                info_text.insert(tk.END, f"  {param}: {value}\n")
        
        # Model-specific information
        info_text.insert(tk.END, "\nModel-Specific Information:\n")
        
        if isinstance(self.model, RandomForestClassifier):
            info_text.insert(tk.END, f"  Number of trees: {self.model.n_estimators}\n")
            info_text.insert(tk.END, f"  Max depth: {self.model.max_depth}\n")
            if hasattr(self.model, "feature_importances_"):
                info_text.insert(tk.END, "\nFeature Importance:\n")
                importance = self.model.feature_importances_
                for i, feat in enumerate(['Age', 'Gender']):
                    info_text.insert(tk.END, f"  {feat}: {importance[i]:.4f}\n")
        
        elif isinstance(self.model, KNeighborsClassifier):
            info_text.insert(tk.END, f"  Number of neighbors: {self.model.n_neighbors}\n")
            info_text.insert(tk.END, f"  Distance metric: {self.model.metric}\n")
        
        elif isinstance(self.model, GaussianNB):
            if hasattr(self.model, "class_prior_"):
                info_text.insert(tk.END, "\nClass Priors:\n")
                for i, cls in enumerate(self.model.classes_):
                    info_text.insert(tk.END, f"  {cls}: {self.model.class_prior_[i]:.4f}\n")
        
        # Make the text widget read-only
        info_text.config(state=tk.DISABLED)
        
        # Close button
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)


def main():
    root = tk.Tk()
    app = MusicPreferencesApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()