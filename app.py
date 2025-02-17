import os
import pandas as pd
import streamlit as st
from groq import Groq

# Load the Groq API Key from Hugging Face Secrets
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # use groq api key, make it from here "https://console.groq.com/keys?_gl=1*gbamid*_ga*MTI3ODQyMDY4Ny4xNzM3ODc0MjIx*_ga_4TD0X2GEZG*MTczOTc3MDI5MS4xMC4wLjE3Mzk3NzAyOTEuMC4wLjA."
client = Groq(api_key=GROQ_API_KEY)

# Function to load data
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# Function to fetch insights based on ID
def get_land_insights(id, data):
    if id not in data['ID'].values:
        return f"**❌ Error:** ID {id} not found in the dataset."

    # Extract the relevant row
    land_data = data[data['ID'] == id].iloc[0]

    # Formulate query for the Groq model
    query = (
        f"Provide insights for the following land:\n"
        f"- Soil Quality: {land_data['Soil_Quality']}\n"
        f"- Seed Variety: {land_data['Seed_Variety']}\n"
        f"- Fertilizer Amount: {land_data['Fertilizer_Amount_kg_per_hectare']} kg/hectare\n"
        f"- Sunny Days: {land_data['Sunny_Days']}\n"
        f"- Rainfall: {land_data['Rainfall_mm']} mm\n"
        f"- Irrigation Schedule: {land_data['Irrigation_Schedule']}\n"
    )

    # Send query to Groq API
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert in agricultural yield prediction."},
                {"role": "user", "content": query},
            ],
            model="llama-3.3-70b-versatile",
        )
        insights = chat_completion.choices[0].message.content

        # Append predicted yield (if available)
        if 'Yield_kg_per_hectare' in land_data:
            insights += f"\n\n**🌾 Predicted Yield:** {land_data['Yield_kg_per_hectare']} kg/hectare."

        return insights

    except Exception as e:
        return f"**❌ Error generating insights:** {str(e)}"


# Streamlit App
def main():
    # App title
    st.title("🌾 **Agricultural Insights and Yield Prediction**")
    st.markdown(
        """
        Welcome to the **Agricultural Insights App**! 🌟 This tool helps you predict crop yields and provides 
        valuable insights about your land based on weather, soil, and agricultural practices. 
        **🌟 Key Features:**
        - Upload your dataset 📁
        - Get insights and recommendations for your land 🔍
        - Understand your agricultural yield predictions 🌾
        """
    )

    # Dataset upload section
    st.markdown("### 📂 **Upload Your Dataset**")
    uploaded_file = st.file_uploader("Upload a CSV file containing land and agricultural data.", type=["csv"])

    if uploaded_file:
        # Load dataset
        data = load_data(uploaded_file)
        st.success("✅ Dataset loaded successfully!")
        
        # Display the dataset
        with st.expander("👀 **View Dataset**"):
            st.dataframe(data)

        # Input for Land ID
        st.markdown("### 🔍 **Get Insights for a Land ID**")
        id_input = st.number_input("Enter Land ID:", min_value=0, step=1, format="%d")

        # Submit button
        if st.button("🔎 Get Insights"):
            with st.spinner("Fetching insights... 🚀"):
                output = get_land_insights(id_input, data)
            st.markdown(output, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please upload a dataset to proceed.")

    # Footer section
    st.markdown("---")
    st.markdown(
        """
        🌟 **Powered by [Groq AI](https://groq.com)**  
        Developed with ❤️ using Streamlit.
        """
    )


# Run the Streamlit app
if __name__ == "__main__":
    main()
