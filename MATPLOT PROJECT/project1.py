
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('DATASET/bike_sales_india.csv')
print(df)

df.info()
print(df.shape)
print(df.duplicated().sum())
print(df.isnull().sum())

print(df.tail())
print(df.head())

                                     #=================#
        #=========================== DATA VISUALIZATION =================#
                                     #=================#   

#SCATTER PLOT
plt.scatter(x=df['State'].head(30),y=df['Price (INR)'].head(30))
plt.xlabel("state")
plt.ylabel("Price(INR)")
plt.title("STATE VS PRICE(INR)")
plt.show()


#BAR PLOT
random_data = df.sample(n=15)
plt.bar(random_data['Model'], random_data['Brand'])
plt.xlabel("MODEL")
plt.ylabel("Brand")
plt.title("MODEL VS Brand")
plt.show()

#HISTOGRAM P
	
plt.hist(df['Engine Capacity (cc)'], bins=10)

plt.xlabel("Engine Capacity (cc)")
plt.ylabel("Frequency")
plt.title("Distribution of Engine Capacity")
plt.show()


#HORIZONTIAL BAR PLOT

plt.barh(
    y=df['Owner Type'].head(),
    width=df['Resale Price (INR)'].head(),
    height=0.5
)

plt.xlabel("Average Resale Price")
plt.ylabel("Owner Type")
plt.title("Owner Type vs Average Resale Price")

plt.show()


#PIE LPOT
brand_counts = df['Brand'].value_counts().head(10)

plt.pie(
    brand_counts,
    labels=brand_counts.index,
    autopct='%1.1f%%'
)

plt.title("Top 10 Brands")
plt.show()