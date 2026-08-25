import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('data.csv')

df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
df = df.drop_duplicates()  

df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
df['bhk_count'] = df['bhk_count'].astype(int)
df['rate_per_sqft'] = df['rate_per_sqft'].str.replace('$', '').str.replace(',', '').astype(int)

print("Duplicates after cleaning:", df.duplicated().sum())  
print("Shape after cleaning:", df.shape)

df.to_csv('cleaned_data.csv', index=False)

fig, axes = plt.subplots(4, 3, figsize=(20, 22))
fig.suptitle('Gurgaon Real Estate Market Analysis', fontsize=20, fontweight='bold')

# 1. Costliest flats
top10 = df.nlargest(10, 'price')
sns.barplot(data=top10, x='price', y='socity', ax=axes[0, 0])
axes[0, 0].set_title('Top 10 Costliest Properties')

# 2. Locality with highest avg price
loc_price = df.groupby('locality')['price'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=loc_price.values, y=loc_price.index, ax=axes[0, 1])
axes[0, 1].set_title('Top Localities by Avg Price')

# 3. Locality with highest rate per sqft
loc_psf = df.groupby('locality')['rate_per_sqft'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=loc_psf.values, y=loc_psf.index, ax=axes[0, 2])
axes[0, 2].set_title('Top Localities by Rate/Sqft')

# 4. Ready-to-move vs Under-construction (status has 4 categories, all shown)
sns.boxplot(data=df, x='status', y='price', ax=axes[1, 0])
axes[1, 0].set_title('Price by Status')
axes[1, 0].tick_params(axis='x', rotation=20)

# 5. RERA approval premium
sns.boxplot(data=df, x='rera_approval', y='price', ax=axes[1, 1])
axes[1, 1].set_title('Price: RERA vs Non-RERA')
axes[1, 1].tick_params(axis='x', rotation=15)

# 6. Area vs Price
sns.scatterplot(data=df, x='area', y='price', ax=axes[1, 2], alpha=0.4)
axes[1, 2].set_title('Area vs Price')

# 7. BHK count vs avg price
bhk_price = df.groupby('bhk_count')['price'].mean().sort_index()
sns.barplot(x=bhk_price.index, y=bhk_price.values, ax=axes[2, 0])
axes[2, 0].set_title('Avg Price by BHK Count')

# 8. Property/flat type vs price (Apartment, Floor, Plot, Villa, House, Penthouse)
sns.boxplot(data=df, x='flat_type', y='price', ax=axes[2, 1])
axes[2, 1].set_title('Price by Flat Type')
axes[2, 1].tick_params(axis='x', rotation=25)

# 9. Top builders by avg price
builder_price = df.groupby('builder_name')['price'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=builder_price.values, y=builder_price.index, ax=axes[2, 2])
axes[2, 2].set_title('Top 10 Builders by Avg Price')

# 10. Area vs rate_per_sqft (larger homes cheaper per sqft?)
sns.scatterplot(data=df, x='area', y='rate_per_sqft', ax=axes[3, 0], alpha=0.4)
axes[3, 0].set_title('Area vs Rate/Sqft')

# 11. Correlation heatmap
numeric_cols = df[['price', 'area', 'rate_per_sqft', 'bhk_count']]
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', ax=axes[3, 1])
axes[3, 1].set_title('Correlation Heatmap')

# 12. Avg rate/sqft by flat type
flat_psf = df.groupby('flat_type')['rate_per_sqft'].mean().sort_values(ascending=False)
sns.barplot(x=flat_psf.values, y=flat_psf.index, ax=axes[3, 2])
axes[3, 2].set_title('Avg Rate/Sqft by Flat Type')

plt.tight_layout()
plt.savefig('gurgaon_analysis_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()