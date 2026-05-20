import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, 'data', 'insurance.csv')
# Set the style for professional presentation graphs
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12})

def generate_model_comparison_graph():
    print("Generating Model Comparison Graph...")
    
    # data
    models = ['Linear\nRegression', 'Random\nForest', 'Gradient\nBoosting']
    r2_scores = [0.5953, 0.8772, 0.8694]
    mae_scores = [3913.33, 1964.47, 2099.53]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    #  Y-axis for R-Squared (Bar Chart)
    color = 'tab:blue'
    ax1.set_ylabel('R² Score (Higher is Better)', color=color, fontweight='bold')
    bars = ax1.bar(models, r2_scores, color=color, alpha=0.7, width=0.4, label='R² Score')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.0)

    # R2 values on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', 
                 ha='center', va='bottom', fontweight='bold')

    # Secondary Y-axis for MAE (Line Chart)
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Mean Absolute Error ($) (Lower is Better)', color=color, fontweight='bold')  
    
    lines = ax2.plot(models, mae_scores, color=color, marker='o', markersize=10, 
                     linewidth=3, label='MAE ($)')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 5000)

    # add MAE values near the points
    for i, txt in enumerate(mae_scores):
        ax2.annotate(f'${txt:,.2f}', (models[i], mae_scores[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center', 
                     fontweight='bold', color=color)
        
    plt.title('Model Comparison Results: Predicting Health Insurance Costs', fontweight='bold', fontsize=14)
    fig.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.close()
    print("Saved: 'model_comparison.png'")

def generate_bmi_smoker_graph():
    print("Generating BMI vs. Smoker Graph...")
    
    data_path = DATA_PATH
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}. Make sure it's in the same folder as this script.")
        return
        
    df = pd.read_csv(data_path)
    
    plt.figure(figsize=(10, 6))
    
    # Red for smokers and Blue for non-smokers
    palette = {"yes": "#e74c3c", "no": "#3498db"} 
    
    sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker', palette=palette, alpha=0.7, s=60)
    
    # obese line
    plt.axvline(x=30, color='gray', linestyle='--', linewidth=2, label='Obesity Threshold (BMI=30)')
 
    plt.text(31, 45000, 'Explosive Cost Increase\n(Smoker + BMI > 30)', 
             color='#c0392b', fontweight='bold', fontsize=11, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='#c0392b', boxstyle='round,pad=0.5'))

    plt.title('Why Linear Regression Fails:\nThe Non-Linear Impact of Smoking & BMI on Costs', 
              fontweight='bold', fontsize=14)
    plt.xlabel('Body Mass Index (BMI)', fontweight='bold')
    plt.ylabel('Insurance Charges ($)', fontweight='bold')

    plt.legend(title='Smoker?', title_fontsize='12', loc='upper left')
    
    plt.tight_layout()
    plt.savefig('smoker_bmi_charges.png', dpi=300)
    plt.close()
    print("Saved: 'smoker_bmi_charges.png'")

if __name__ == '__main__':
    generate_model_comparison_graph()
    generate_bmi_smoker_graph()