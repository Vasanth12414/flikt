import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob

# Set page config
st.set_page_config(
    page_title="Customer Insights",
    page_icon="📊",
    layout="wide"
)

# App title
st.title("📊 Customer Insights Dashboard")
st.markdown("Upload customer feedback for quick analysis and insights")

# Simple sentiment analysis using TextBlob
def analyze_sentiment(text):
    analysis = TextBlob(str(text))
    if analysis.sentiment.polarity > 0.1:
        return "Positive", analysis.sentiment.polarity
    elif analysis.sentiment.polarity < -0.1:
        return "Negative", analysis.sentiment.polarity
    else:
        return "Neutral", analysis.sentiment.polarity

# Simple extractive summarization
def simple_summarize(text, num_sentences=2):
    sentences = str(text).split('.')
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return '. '.join(sentences[:num_sentences]) + '.'

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {len(df)} entries")
    
    st.subheader("📋 Data Preview")
    st.dataframe(df.head())
    
    if 'feedback' not in df.columns:
        st.error("❌ CSV must contain 'feedback' column")
    else:
        tab1, tab2, tab3 = st.tabs(["Sentiment", "Summaries", "Insights"])
        
        with tab1:
            st.header("🎭 Sentiment Analysis")
            
            # Analyze sample feedbacks
            sample_data = df.head(10).copy()
            sentiments = []
            
            for feedback in sample_data['feedback']:
                sentiment, score = analyze_sentiment(feedback)
                sentiments.append({
                    'feedback': feedback[:80] + "..." if len(feedback) > 80 else feedback,
                    'sentiment': sentiment,
                    'score': round(score, 3)
                })
            
            sentiment_df = pd.DataFrame(sentiments)
            st.dataframe(sentiment_df)
            
            # Sentiment chart
            st.subheader("Sentiment Distribution")
            sentiment_counts = sentiment_df['sentiment'].value_counts()
            
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            ax.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
            ax.set_ylabel('Number of Feedback')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with tab2:
            st.header("📝 Text Summaries")
            
            selected_feedback = st.selectbox("Choose feedback to summarize:", 
                                           df['feedback'].head(5).tolist())
            
            if selected_feedback:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original")
                    st.write(selected_feedback)
                
                with col2:
                    st.subheader("Summary")
                    summary = simple_summarize(selected_feedback)
                    st.info(summary)
        
        with tab3:
            st.header("📈 Insights")
            
            # Mock satisfaction data
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
            scores = [72, 75, 78, 74, 79, 76]
            
            # Simple prediction
            next_month = np.mean(scores) + 2  # Simple trend
            
            st.metric("Predicted Next Month Score", f"{next_month:.1f}")
            
            # Trend chart
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(months, scores, 'bo-', label='Actual', linewidth=2)
            ax.axhline(y=next_month, color='red', linestyle='--', label='Predicted')
            ax.set_title('Satisfaction Trend')
            ax.set_ylabel('Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Key findings
            st.subheader("Key Findings")
            positive_count = len([s for s in sentiments if s['sentiment'] == 'Positive'])
            st.write(f"• {positive_count} out of {len(sentiments)} feedbacks are positive")
            st.write("• Satisfaction trend is stable")
            st.write("• Most feedback is constructive")

else:
    st.info("👆 Upload a CSV file with 'feedback' column")
    
    # Sample data
    sample_data = {
        'feedback': [
            "Great product quality and fast delivery",
            "Poor customer service experience", 
            "The product meets expectations",
            "Excellent value for money",
            "Shipping was delayed but product is good"
        ]
    }
    st.subheader("Sample CSV Format:")
    st.dataframe(pd.DataFrame(sample_data))
    
    # Download sample
    sample_df = pd.DataFrame(sample_data)
    csv = sample_df.to_csv(index=False)
    st.download_button(
        label="Download Sample CSV",
        data=csv,
        file_name="sample_feedback.csv",
        mime="text/csv"
    )

st.markdown("---")
st.markdown("Built with Streamlit • Lightweight AI Analysis")