import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

def generate_advanced_music_dataset(output_file='synthetic_music_data.csv', num_rows=2500, visualize=True):
    """
    Generate a synthetic music dataset with 2,500 lines based on patterns observed
    in the original dataset, with options for visualization and analysis.
    
    Parameters:
    -----------
    output_file : str
        Filename for the output CSV file
    num_rows : int
        Number of rows to generate
    visualize : bool
        Whether to generate visualizations of the data
    
    Returns:
    --------
    pandas.DataFrame
        The generated synthetic dataset
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Original data insights
    # Age: range 20-37, mean ~28
    # Gender: 0 and 1 (binary), evenly distributed
    # Genres: Classical (33%), HipHop, Jazz, Dance, Acoustic (16.7% each)
    # Patterns: 
    # - Gender 1: HipHop (young), Jazz (middle), Classical (older)
    # - Gender 0: Dance (young), Acoustic (middle), Classical (older)
    
    # Create empty DataFrame
    df = pd.DataFrame(columns=['age', 'gender', 'genre'])
    
    # Generate synthetic data
    for i in range(num_rows):
        # Generate gender (approximately 50/50 split)
        gender = random.randint(0, 1)
        
        # Generate age with normal-like distribution
        if gender == 1:
            # Slightly older age distribution for gender 1
            age = np.random.normal(29, 5)
        else:
            # Slightly younger age distribution for gender 0
            age = np.random.normal(27, 5)
        
        # Ensure age is within original bounds (20-37)
        age = max(20, min(37, int(round(age))))
        
        # Generate genre based on gender and age (with some randomness)
        rand_factor = random.random()
        
        if gender == 1:  # Gender 1 patterns
            if age < 26:
                genre = "HipHop" if rand_factor < 0.8 else random.choice(["Jazz", "Classical", "Dance", "Acoustic"])
            elif age < 31:
                genre = "Jazz" if rand_factor < 0.8 else random.choice(["HipHop", "Classical", "Dance", "Acoustic"])
            else:
                genre = "Classical" if rand_factor < 0.8 else random.choice(["HipHop", "Jazz", "Dance", "Acoustic"])
        else:  # Gender 0 patterns
            if age < 26:
                genre = "Dance" if rand_factor < 0.8 else random.choice(["HipHop", "Jazz", "Classical", "Acoustic"])
            elif age < 31:
                genre = "Acoustic" if rand_factor < 0.8 else random.choice(["HipHop", "Jazz", "Classical", "Dance"])
            else:
                genre = "Classical" if rand_factor < 0.8 else random.choice(["HipHop", "Jazz", "Dance", "Acoustic"])
        
        # Add row to DataFrame
        df.loc[i] = [age, gender, genre]
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    # Print summary statistics
    print(f"Generated {num_rows} rows of synthetic music data")
    print(f"Saved to: {output_file}")
    
    print("\nData Summary:")
    print(f"Age range: {df['age'].min()}-{df['age'].max()}, Average: {df['age'].mean():.2f}")
    
    gender_counts = df['gender'].value_counts(normalize=True) * 100
    print(f"\nGender distribution:")
    for gender, percentage in gender_counts.items():
        print(f"Gender {gender}: {percentage:.2f}%")
    
    genre_counts = df['genre'].value_counts(normalize=True) * 100
    print(f"\nGenre distribution:")
    for genre, percentage in genre_counts.items():
        print(f"{genre}: {percentage:.2f}%")
    
    # Create visualizations if requested
    if visualize:
        # Set up the visualization style
        sns.set(style="whitegrid")
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Age distribution
        plt.subplot(2, 2, 1)
        sns.histplot(data=df, x='age', hue='gender', multiple='stack', bins=range(20, 38))
        plt.title('Age Distribution by Gender')
        plt.xlabel('Age')
        plt.ylabel('Count')
        
        # Plot 2: Genre distribution
        plt.subplot(2, 2, 2)
        sns.countplot(data=df, x='genre', order=df['genre'].value_counts().index)
        plt.title('Genre Distribution')
        plt.xlabel('Genre')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Plot 3: Genre by Gender
        plt.subplot(2, 2, 3)
        genre_gender = pd.crosstab(df['genre'], df['gender'], normalize='columns') * 100
        genre_gender.plot(kind='bar', stacked=False)
        plt.title('Genre Preference by Gender (%)')
        plt.xlabel('Genre')
        plt.ylabel('Percentage')
        plt.xticks(rotation=45)
        plt.legend(['Gender 0', 'Gender 1'])
        
        # Plot 4: Age vs. Genre
        plt.subplot(2, 2, 4)
        sns.boxplot(data=df, x='genre', y='age')
        plt.title('Age Distribution by Genre')
        plt.xlabel('Genre')
        plt.ylabel('Age')
        plt.xticks(rotation=45)
        
        # Adjust layout and save figure
        plt.tight_layout()
        plt.savefig('music_data_visualization.png')
        plt.close()
        
        print("\nVisualizations saved to 'music_data_visualization.png'")
    
    return df

# Example usage
if __name__ == "__main__":
    # Generate 2,500 rows with visualizations
    df = generate_advanced_music_dataset()
    
    # Display correlation between age, gender, and genre
    print("\nAnalyzing correlations in the synthetic data:")
    
    # Age by genre
    print("\nAverage age by genre:")
    print(df.groupby('genre')['age'].mean().sort_values())
    
    # Crosstab of gender and genre
    print("\nGender-Genre distribution (counts):")
    gender_genre_counts = pd.crosstab(df['gender'], df['genre'])
    print(gender_genre_counts)
    
    print("\nGender-Genre distribution (percentages within gender):")
    gender_genre_pct = pd.crosstab(df['gender'], df['genre'], normalize='index') * 100
    print(gender_genre_pct.round(2))
