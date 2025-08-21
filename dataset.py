import pandas as pd
import numpy as np

# Generate 1000 random years of experience between 0.5 and 20
np.random.seed(42)
years_experience = np.random.uniform(0.5, 20, 1000)

# Generate salaries with a base slope + some random noise
salary = 30000 + (years_experience * 9000) + np.random.normal(0, 5000, 1000)

# Create DataFrame
data = pd.DataFrame({
    'YearsExperience': years_experience,
    'Salary': salary
})

# Save to CSV
data.to_csv('Salary_Data.csv', index=False)

print("Salary_Data.csv created with 1000 rows.")
