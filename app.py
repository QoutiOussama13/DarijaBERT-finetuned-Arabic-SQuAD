import streamlit as st
from transformers import pipeline
import huggingface_hub

def main():
    st.set_page_config(page_title="Question Answering Bot", page_icon=":question:")
    st.title("Question Answering Bot")
    
    description = """
    هذا هو بوت بسيط للإجابة على الأسئلة يستخدم نموذجًا مدربًا مسبقًا للإجابة على الأسئلة استنادًا إلى السياق المعطى.
    أدخل السياق والسؤال في المربعات المناسبة وانقر على زر "اطرح سؤال" للحصول على الإجابة.
    """
    st.markdown(description)
    
    st.subheader(": أدخل السياق")
    context = st.text_area(" : السياق", max_chars=1000, key="context")
    
    st.subheader(" : أدخل السؤال ")
    question = st.text_input("السؤال", max_chars=200, key="question")
    
    if st.button("اطرح سؤال"):
        if context and question:
            with st.spinner("جاري المعالجة..."):
                question_answerer = pipeline("question-answering", model="JasperV13/my_qa_model")
                result = question_answerer(context=context, question=question)
                st.success("تم إنشاء الإجابة!")
                st.write("**الإجابة**: ", result['answer'])
                st.write("**النتيجة**: ", result['score'])
                st.write("**بداية الفهرس**: ", result['start'])
                st.write("**نهاية الفهرس**: ", result['end'])
        else:
            st.error("يرجى إدخال كلاً من السياق والسؤال.")
    

if __name__ == '__main__':
    main()
