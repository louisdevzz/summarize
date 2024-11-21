import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Tải mô hình và tokenizer từ Hugging Face
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base-vietnamese-summarization")
    model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base-vietnamese-summarization")
    return tokenizer, model

# Hàm xử lý văn bản dài
def summarize_long_text(text, tokenizer, model):
    # Chia văn bản thành các đoạn 512 token
    chunk_size = 512
    tokens = tokenizer.encode(text, return_tensors="pt", truncation=False)
    chunks = [tokens[:, i:i + chunk_size] for i in range(0, tokens.shape[1], chunk_size)]

    # Tóm tắt từng đoạn
    summaries = []
    for chunk in chunks:
        output = model.generate(
            chunk,
            max_length=150,
            min_length=50,
            num_beams=8,
            length_penalty=1.5,
            early_stopping=True
        )
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        summaries.append(summary)
    
    # Kết hợp các đoạn tóm tắt
    return " ".join(summaries)

# Giao diện Streamlit
st.title("Vietnamese Text Summarization")
st.write("Ứng dụng tóm tắt nội dung tiếng Việt sử dụng mô hình AI mạnh mẽ từ Hugging Face.")

# Nhập văn bản cần tóm tắt
input_text = st.text_area("Nhập văn bản của bạn:", height=200)

# Xử lý tóm tắt khi người dùng nhấn nút
if st.button("Tóm tắt"):
    if input_text.strip():
        st.write("Đang xử lý... Vui lòng đợi!")
        tokenizer, model = load_model()
        summary = summarize_long_text(input_text, tokenizer, model)
        st.subheader("Kết quả tóm tắt:")
        st.write(summary)
    else:
        st.warning("Vui lòng nhập văn bản trước khi tóm tắt.")

# Footer
st.markdown("---")
st.markdown("Ứng dụng sử dụng mô hình [VietAI/vit5-base-vietnamese-summarization](https://huggingface.co/VietAI/vit5-base-vietnamese-summarization).")
