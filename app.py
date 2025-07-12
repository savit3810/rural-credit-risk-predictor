import streamlit as st
import pandas as pd
import joblib
import os
import hashlib
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import altair as alt
from fpdf import FPDF
from openai import OpenAI

# OpenAI API Key - Direct configuration
OPENAI_API_KEY = "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"

USERS_FILE = "users.csv"

# --- Password Hashing ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- Load or Initialize Users File ---
def load_users():
    if os.path.exists(USERS_FILE):
        return pd.read_csv(USERS_FILE)
    else:
        df = pd.DataFrame(columns=['username', 'password'])
        df.to_csv(USERS_FILE, index=False)
        return df

# --- Save New User ---
def save_user(username, password):
    df = load_users()
    if username in df['username'].values:
        return False
    hashed_password = hash_password(password)
    new_user = pd.DataFrame([[username, hashed_password]], columns=['username', 'password'])
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(USERS_FILE, index=False)
    return True

# --- Password Validation ---
def is_valid_password(password):
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    if not re.match("^[a-zA-Z0-9_]+$", password):
        return False, "Password can only contain letters, numbers, and underscores"
    return True, ""

# --- Login and Signup System ---
def auth_system():
    st.set_page_config(page_title="Rural Credit Risk Predictor", page_icon="🌾", layout="wide")
    
    # Check if logo exists, if not skip it
    logo_path = "C:/Users/Lenovo/Pictures/logo_agriculture.png"
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=80)
    else:
        st.sidebar.write("🌾")  # Fallback emoji
    
    st.sidebar.title("🌾 Credit Predictor")
    st.sidebar.markdown("---")

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    auth_mode = st.sidebar.radio("Choose option", ["🔐 Login", "🆕 Signup"])

    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if auth_mode == "🔐 Login":
        if st.sidebar.button("Login"):
            users_df = load_users()
            hashed_input_pw = hash_password(password)
            if ((users_df['username'] == username) & (users_df['password'] == hashed_input_pw)).any():
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.sidebar.error("❌ Invalid username or password")

    elif auth_mode == "🆕 Signup":
        if st.sidebar.button("Create Account"):
            if username.strip() == "" or password.strip() == "":
                st.sidebar.warning("⚠️ Username and password cannot be empty")
            else:
                valid_pw, msg = is_valid_password(password)
                if not valid_pw:
                    st.sidebar.warning(f"⚠️ {msg}")
                elif save_user(username, password):
                    st.sidebar.success("✅ Account created! Please login.")
                else:
                    st.sidebar.warning("⚠️ Username already exists")

    if not st.session_state.logged_in:
        # Check if logo exists for main page
        if os.path.exists(logo_path):
            st.image(logo_path, width=250)
        else:
            st.markdown("# 🌾")  # Fallback emoji
        st.markdown("""
### 🌾 Welcome to the Rural Credit Risk Predictor
Please login or sign up to continue.
""")
        st.stop()

# --- Call Auth System ---
auth_system()

# --- Logout Button ---
if st.session_state.logged_in:
    with st.sidebar:
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()

# --- OpenAI API Integration ---
def generate_gpt_response(prompt):
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a loan officer assistant that explains credit decisions in simple language."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error: {e}"

# --- Load Model ---
MODEL_PATH = "credit_model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.error("🚫 Model file 'credit_model.pkl' not found. Please train the model first.")
    st.stop()

# --- App Layout ---
st.title("🌾 Rural Credit Risk Predictor")
st.markdown("## 💡 Predict rural loan eligibility using ML + charts")
st.markdown("---")

# --- Main Tabs ---
tabs = st.tabs(["📋 Single Prediction", "📂 Bulk CSV Upload", "📈 Charts", "📊 Model Metrics", "🤖 AI Assistant"])

# --- Tab 1: Single Prediction ---
with tabs[0]:
    with st.form("credit_form"):
        st.subheader("📋 Enter Applicant Details")
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=70, value=35)
            income = st.number_input("Monthly Income (₹)", min_value=0, value=30000)
            land = st.number_input("Land Owned (acres)", min_value=0.0, value=2.5)
            rainfall = st.number_input("Last Year Rainfall (mm)", min_value=0, value=800)

        with col2:
            crop = st.selectbox("Crop Type", ["wheat", "rice", "sugarcane"])
            price = st.number_input("Market Price (₹/quintal)", min_value=0, value=1600)
            upi_score = st.slider("UPI Usage Score", 0.0, 1.0, value=0.6)

        submitted = st.form_submit_button("🚀 Predict")

    if submitted:
        st.subheader("📊 Prediction Result")
        crop_encoded = {
            "crop_type_rice": 1 if crop == "rice" else 0,
            "crop_type_sugarcane": 1 if crop == "sugarcane" else 0,
            "crop_type_wheat": 1 if crop == "wheat" else 0
        }
        input_df = pd.DataFrame([{ 
            "age": age, 
            "income": income, 
            "land_area": land, 
            "rainfall_last_year": rainfall,
            "market_price": price, 
            "upi_usage_score": upi_score, 
            **crop_encoded 
        }])
        
        try:
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]
            
            if pred == 1:
                st.success(f"✅ Eligible for Loan (Confidence: {prob:.2f})")
            else:
                st.error(f"❌ Not Eligible for Loan (Confidence: {prob:.2f})")

            # --- Generate GPT Explanation ---
            prompt = f"""
            A farmer has income ₹{income}/month, owns {land} acres, grows {crop}, had {rainfall}mm rainfall, market price ₹{price}, and UPI score {upi_score}.
            Explain in 1-2 lines why this farmer is {'eligible' if pred == 1 else 'not eligible'} for a loan.
            """
            
            try:
                explanation = generate_gpt_response(prompt)
                if explanation.startswith("⚠️ Error"):
                    raise Exception("API failed")
            except:
                # --- Mock fallback explanation ---
                if pred == 1:
                    explanation = "Based on your income and land size, your profile looks promising for loan eligibility."
                else:
                    explanation = "Low digital activity and limited land/income may affect your eligibility."
            
            st.info(f"🤖 AI Explanation: {explanation}")

            # --- Generate AI Advice Letter (only if not eligible) ---
            if pred == 0:
                advice_prompt = f"""
                A rural applicant is not eligible for a loan based on the following profile:
                - Age: {age}
                - Income: ₹{income}/month
                - Land: {land} acres
                - Crop: {crop}
                - Market Price: ₹{price}
                - Rainfall: {rainfall}mm
                - UPI Score: {upi_score}
                
                Suggest 2-3 actionable improvements they can take to become eligible in simple, polite language.
                """
                
                try:
                    advice = generate_gpt_response(advice_prompt)
                    if advice.startswith("⚠️ Error"):
                        raise Exception("API failed")
                except:
                    advice = "Consider increasing your income or land size, and improve UPI usage. Also check government subsidy programs."
                
                st.warning(f"📄 Advice Letter: {advice}")

                # --- Format Advice Letter Text ---
                # Clean the advice text to remove special characters that might cause PDF issues
                clean_advice = advice.replace('₹', 'Rs.').replace('📌', '*').replace('✅', '*').replace('⚠️', '*')
                
                full_letter = f"""
Rural Credit Risk Advice Letter
----------------------------------

Dear Applicant,

Based on your current profile, you are currently not eligible for a rural loan.

* Recommendation:
{clean_advice}

Please visit your nearest rural bank or agriculture center for further assistance.

Sincerely,
AI Credit Officer
"""

                # --- Generate Downloadable PDF ---
                try:
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_auto_page_break(auto=True, margin=15)
                    pdf.set_font("Arial", size=12)
                    
                    # Clean each line to remove problematic characters
                    for line in full_letter.split('\n'):
                        # Remove or replace problematic Unicode characters
                        clean_line = line.strip()
                        clean_line = clean_line.encode('latin-1', 'ignore').decode('latin-1')
                        if clean_line:  # Only add non-empty lines
                            pdf.cell(200, 10, txt=clean_line, ln=True)
                    
                    pdf_output = "advice_letter.pdf"
                    pdf.output(pdf_output)

                    with open(pdf_output, "rb") as f:
                        st.download_button(
                            label="Download Advice Letter as PDF",
                            data=f,
                            file_name="loan_advice_letter.pdf",
                            mime="application/pdf"
                        )
                except Exception as pdf_error:
                    st.error(f"PDF generation error: {pdf_error}")
                    # Provide alternative text download
                    st.download_button(
                        label="Download Advice Letter as Text",
                        data=full_letter,
                        file_name="loan_advice_letter.txt",
                        mime="text/plain"
                    )
                    
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# --- Tab 2: CSV Upload ---
with tabs[1]:
    st.subheader("📂 Bulk Prediction via CSV")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        try:
            df_csv = pd.read_csv(uploaded_file)
            expected_cols = ['age', 'income', 'land_area', 'rainfall_last_year', 'market_price', 'upi_usage_score',
                             'crop_type_rice', 'crop_type_sugarcane', 'crop_type_wheat']
            
            if all(col in df_csv.columns for col in expected_cols):
                preds = model.predict(df_csv[expected_cols])
                probs = model.predict_proba(df_csv[expected_cols])[:, 1]
                df_csv['Predicted_Loan_Eligible'] = preds
                df_csv['Confidence_Score'] = probs.round(2)
                
                # Store in session state for use in other tabs
                st.session_state.df_csv = df_csv
                
                st.success("✅ Predictions completed!")
                st.dataframe(df_csv.head())
                
                csv_download = df_csv.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Results CSV", 
                    data=csv_download, 
                    file_name="loan_predictions.csv", 
                    mime='text/csv'
                )
            else:
                st.warning("⚠️ Your CSV must include: " + ", ".join(expected_cols))
        except Exception as e:
            st.error(f"CSV Processing Error: {e}")

# --- Tab 3: Charts ---
with tabs[2]:
    st.subheader("📊 Visual Insights from Uploaded Data")
    
    # Check if data exists in session state
    if 'df_csv' in st.session_state:
        df_csv = st.session_state.df_csv
        try:
            st.markdown("#### 🧠 UPI Score vs Approval Rate")
            df_csv['upi_bin'] = pd.cut(df_csv['upi_usage_score'], bins=5).astype(str)
            chart_data = df_csv.groupby('upi_bin')['Predicted_Loan_Eligible'].mean().reset_index()
            chart = alt.Chart(chart_data).mark_bar().encode(
                x='upi_bin:N', 
                y='Predicted_Loan_Eligible:Q', 
                tooltip=['upi_bin', 'Predicted_Loan_Eligible']
            ).properties(width=600)
            st.altair_chart(chart, use_container_width=True)

            st.markdown("#### 👤 Age Distribution")
            age_chart = alt.Chart(df_csv).mark_bar().encode(
                x=alt.X('age:Q', bin=True), 
                y='count()', 
                tooltip=['age']
            ).properties(width=600)
            st.altair_chart(age_chart, use_container_width=True)

            st.markdown("#### 🌾 Crop Type vs Approval Rate")
            crop_cols = ['crop_type_rice', 'crop_type_sugarcane', 'crop_type_wheat']
            crop_data = { 'Crop': [], 'Approval Rate': [] }
            for crop in crop_cols:
                crop_name = crop.replace('crop_type_', '').capitalize()
                crop_subset = df_csv[df_csv[crop] == 1]
                if len(crop_subset) > 0:
                    rate = crop_subset['Predicted_Loan_Eligible'].mean()
                    crop_data['Crop'].append(crop_name)
                    crop_data['Approval Rate'].append(round(rate, 2))
            
            if crop_data['Crop']:  # Only create chart if there's data
                crop_df = pd.DataFrame(crop_data)
                crop_chart = alt.Chart(crop_df).mark_bar().encode(
                    x='Crop', 
                    y='Approval Rate', 
                    tooltip=['Crop', 'Approval Rate']
                ).properties(width=500)
                st.altair_chart(crop_chart, use_container_width=True)

        except Exception as e:
            st.error(f"Chart Error: {e}")
    else:
        st.info("ℹ️ Upload a CSV file first to view insights.")

# --- Tab 4: Metrics ---
with tabs[3]:
    st.subheader("📈 Model Evaluation Metrics")
    
    if 'df_csv' in st.session_state:
        df_csv = st.session_state.df_csv
        if 'Loan_Eligible' in df_csv.columns:
            try:
                y_true = df_csv['Loan_Eligible']
                y_pred = df_csv['Predicted_Loan_Eligible']
                
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎯 Accuracy", f"{acc:.2f}")
                with col2:
                    st.metric("📌 Precision", f"{prec:.2f}")
                with col3:
                    st.metric("📊 Recall", f"{rec:.2f}")
                with col4:
                    st.metric("🏁 F1 Score", f"{f1:.2f}")

                if st.checkbox("Show Confusion Matrix"):
                    cm = confusion_matrix(y_true, y_pred)
                    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
                    st.write(cm_df)
            except Exception as e:
                st.error(f"Metric Error: {e}")
        else:
            st.info("ℹ️ Upload a labeled CSV with 'Loan_Eligible' column to view metrics.")
    else:
        st.info("ℹ️ Upload a CSV file first to view metrics.")

# --- Tab 5: AI Chatbot Assistant ---
with tabs[4]:
    st.subheader("🤖 AI Chatbot Assistant")
    st.markdown("Ask questions about rural credit, loans, agriculture, or get help with the application!")
    
    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hello! I'm your AI assistant for rural credit and agriculture queries. How can I help you today?"}
        ]
    
    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about rural credit, loans, or agriculture..."):
        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Enhanced system prompt for rural credit domain
                    system_prompt = """
                    You are a helpful AI assistant specialized in rural credit, agriculture, and loan applications. 
                    You help farmers and rural applicants understand:
                    - Loan eligibility criteria
                    - Credit improvement strategies
                    - Agricultural best practices
                    - Financial planning for farmers
                    - Government schemes and subsidies
                    - Digital banking and UPI usage
                    
                    Provide practical, actionable advice in simple language. Always be encouraging and supportive.
                    If asked about specific loan predictions, remind users to use the prediction tool in the app.
                    """
                    
                    full_prompt = f"{system_prompt}\n\nUser question: {prompt}"
                    response = generate_gpt_response(full_prompt)
                    
                    # Fallback responses if API fails
                    if response.startswith("⚠️ Error"):
                        fallback_responses = {
                            "loan eligibility": "To improve loan eligibility, focus on: 1) Increasing steady income, 2) Owning more land, 3) Using digital payments regularly, 4) Maintaining good credit history, 5) Choosing profitable crops.",
                            "credit score": "Build credit by: 1) Paying bills on time, 2) Using UPI/digital payments, 3) Keeping loan amounts reasonable, 4) Avoiding multiple loan applications, 5) Maintaining bank account regularly.",
                            "agriculture": "Good agricultural practices include: 1) Crop rotation, 2) Proper irrigation, 3) Quality seeds, 4) Timely fertilization, 5) Pest management, 6) Market price monitoring.",
                            "government schemes": "Check these schemes: 1) PM-KISAN, 2) Crop insurance, 3) Soil health card, 4) Agri-loans at subsidized rates, 5) Rural employment programs. Visit your nearest agriculture office.",
                            "digital banking": "Benefits of digital banking: 1) Easy transactions, 2) Better credit history, 3) Government scheme access, 4) Lower costs, 5) Improved loan eligibility."
                        }
                        
                        # Simple keyword matching for fallback
                        response = "I'd be happy to help! Here are some general tips for rural credit and agriculture:"
                        for keyword, fallback in fallback_responses.items():
                            if keyword in prompt.lower():
                                response = fallback
                                break
                        
                        if response == "I'd be happy to help! Here are some general tips for rural credit and agriculture:":
                            response += "\n\n• Use the prediction tool to check loan eligibility\n• Focus on increasing income and land ownership\n• Improve digital payment usage\n• Maintain good banking relationships\n• Explore government agricultural schemes"
                    
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_response = f"I apologize, but I'm having trouble processing your request. Please try again or use the other features of the app. Error: {str(e)}"
                    st.markdown(error_response)
                    st.session_state.chat_messages.append({"role": "assistant", "content": error_response})
    
    # Quick action buttons
    st.markdown("---")
    st.markdown("**Quick Help:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💡 Loan Tips"):
            tip_response = "Key tips for loan approval: 1) Maintain steady income, 2) Use digital payments regularly, 3) Keep good bank records, 4) Choose profitable crops, 5) Build relationships with local banks."
            st.session_state.chat_messages.append({"role": "user", "content": "Give me loan tips"})
            st.session_state.chat_messages.append({"role": "assistant", "content": tip_response})
            st.rerun()
    
    with col2:
        if st.button("🌾 Crop Advice"):
            crop_response = "Choose crops based on: 1) Local climate and soil, 2) Market demand, 3) Water availability, 4) Your experience, 5) Government support schemes. Popular profitable crops include wheat, rice, and sugarcane in suitable regions."
            st.session_state.chat_messages.append({"role": "user", "content": "Give me crop advice"})
            st.session_state.chat_messages.append({"role": "assistant", "content": crop_response})
            st.rerun()
    
    with col3:
        if st.button("📱 Digital Banking"):
            digital_response = "Benefits of digital banking for farmers: 1) Easy government scheme access, 2) Better loan eligibility, 3) Lower transaction costs, 4) Improved credit history, 5) Convenient payments. Start with UPI apps like PayTM, PhonePe, or Google Pay."
            st.session_state.chat_messages.append({"role": "user", "content": "Tell me about digital banking"})
            st.session_state.chat_messages.append({"role": "assistant", "content": digital_response})
            st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hello! I'm your AI assistant for rural credit and agriculture queries. How can I help you today?"}
        ]
        st.rerun()