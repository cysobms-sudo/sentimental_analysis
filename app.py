# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# -*- coding: utf-8 -*-
"""
تحلیل‌گر احساسات پیشرفته با مدل albert-fa-base-v2
"""

import gradio as gr
from transformers import pipeline
import pandas as pd
import tempfile

print("🧠 در حال بارگیری مدل تحلیل احساسات مالی فارسی...")
print("مدل: m3hrdadfi/albert-fa-base-v2-sentiment-deepsentipers-multi")

pipe = pipeline(
    "text-classification",
    model="m3hrdadfi/albert-fa-base-v2-sentiment-deepsentipers-multi"
)
print("✅ مدل مالی با موفقیت بارگیری شد!")

# ============================================
# CSS برای فونت درشت‌تر
# ============================================
CUSTOM_CSS = """
/* فونت بزرگتر برای همه */
* {
    font-size: 18px !important;
}

h1 {
    font-size: 48px !important;
    font-weight: 900 !important;
    text-align: center !important;
}

h2, h3 {
    font-size: 32px !important;
    font-weight: 700 !important;
}

label, .label-text {
    font-size: 22px !important;
    font-weight: 700 !important;
}

.gr-textbox textarea {
    font-size: 24px !important;
    line-height: 2 !important;
    padding: 20px !important;
}

.gr-button {
    font-size: 24px !important;
    padding: 16px 40px !important;
}

.big-result {
    font-size: 42px !important;
    font-weight: 900 !important;
    text-align: center !important;
    padding: 30px !important;
    border-radius: 20px !important;
    margin: 20px 0 !important;
}

.big-result-positive {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    color: white !important;
    font-size: 44px !important;
    font-weight: 900 !important;
    padding: 35px !important;
    border-radius: 20px !important;
    text-align: center !important;
    box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3) !important;
}

.big-result-negative {
    background: linear-gradient(135deg, #ef4444, #dc2626) !important;
    color: white !important;
    font-size: 44px !important;
    font-weight: 900 !important;
    padding: 35px !important;
    border-radius: 20px !important;
    text-align: center !important;
    box-shadow: 0 10px 30px rgba(239, 68, 68, 0.3) !important;
}

.big-result-neutral {
    background: linear-gradient(135deg, #f59e0b, #d97706) !important;
    color: white !important;
    font-size: 44px !important;
    font-weight: 900 !important;
    padding: 35px !important;
    border-radius: 20px !important;
    text-align: center !important;
    box-shadow: 0 10px 30px rgba(245, 158, 11, 0.3) !important;
}

.big-result-default {
    background: linear-gradient(135deg, #4f8cf7, #6366f1) !important;
    color: white !important;
    font-size: 44px !important;
    font-weight: 900 !important;
    padding: 35px !important;
    border-radius: 20px !important;
    text-align: center !important;
    box-shadow: 0 10px 30px rgba(79, 140, 247, 0.3) !important;
}

.gr-dataframe {
    font-size: 20px !important;
}
.gr-dataframe th {
    font-size: 22px !important;
    font-weight: 700 !important;
}
.gr-markdown {
    font-size: 22px !important;
    line-height: 2 !important;
}
.gr-json {
    font-size: 22px !important;
}
.gr-examples button {
    font-size: 20px !important;
    padding: 12px 24px !important;
}
"""

# ============================================
# تحلیل متن مالی
# ============================================
def analyze_financial_text(text):
    if not text or text.strip() == "":
        return {
            "نتیجه": "⚠️ خطا",
            "درصد اطمینان": "۰%",
            "تعداد کلمات": 0,
            "تعداد کاراکترها": 0,
            "امتیاز": 0,
            "توصیه": "-"
        }
    
    result = pipe(text)[0]
    label = result['label']
    score = result['score']
    
    label_map = {
        'POSITIVE': '📈 صعودی (مثبت)',
        'NEGATIVE': '📉 نزولی (منفی)',
        'NEUTRAL': '➖ خنثی'
    }
    
    advice = {
        'POSITIVE': '✅ خرید / افزایش سرمایه‌گذاری',
        'NEGATIVE': '⚠️ فروش / کاهش سرمایه‌گذاری',
        'NEUTRAL': '⏸️ صبر / بررسی بیشتر'
    }
    
    return {
        "نتیجه": label_map.get(label, label),
        "درصد اطمینان": f"{round(score * 100, 2)}%",
        "تعداد کلمات": len(text.split()),
        "تعداد کاراکترها": len(text),
        "امتیاز": round(score, 4),
        "توصیه": advice.get(label, '⏸️ بررسی بیشتر')
    }

# ============================================
# تحلیل فایل مالی
# ============================================
def analyze_financial_file(file_obj):
    if file_obj is None:
        return "⚠️ لطفاً یک فایل انتخاب کنید.", None, None
    
    try:
        with open(file_obj.name, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            return "⚠️ فایل خالی است.", None, None
        
        results = []
        for idx, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            result = analyze_financial_text(line)
            results.append({
                'خط': idx,
                'متن': line[:150] + ('...' if len(line) > 150 else ''),
                'نتیجه': result['نتیجه'],
                'امتیاز': result['امتیاز'],
                'درصد اطمینان': result['درصد اطمینان'],
                'توصیه': result['توصیه']
            })
        
        if not results:
            return "⚠️ هیچ خط معتبری در فایل یافت نشد.", None, None
        
        df = pd.DataFrame(results)
        temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        df.to_csv(temp_csv.name, index=False, encoding='utf-8-sig')
        
        summary = f"📊 **خلاصه تحلیل اخبار مالی:**\n\n"
        summary += f"📄 **تعداد اخبار تحلیل شده:** {len(df)}\n\n"
        
        label_counts = df['نتیجه'].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            summary += f"- {label}: {count} خبر ({percentage:.1f}%)\n"
        
        positive = len(df[df['نتیجه'].str.contains('صعودی')])
        negative = len(df[df['نتیجه'].str.contains('نزولی')])
        
        if positive > negative:
            market = "📈 **احساسات کلی بازار: صعودی (مثبت)**"
        elif negative > positive:
            market = "📉 **احساسات کلی بازار: نزولی (منفی)**"
        else:
            market = "➖ **احساسات کلی بازار: خنثی**"
        
        summary += f"\n\n{market}"
        
        sample_df = df[['خط', 'متن', 'نتیجه', 'امتیاز', 'توصیه']].head(10)
        
        return summary, temp_csv.name, sample_df
        
    except Exception as e:
        return f"❌ خطا: {str(e)}", None, None

# ============================================
# رابط کاربری
# ============================================
with gr.Blocks(
    title="تحلیلگر احساسات بازار مالی",
    css=CUSTOM_CSS
) as demo:
    
    gr.Markdown("""
    # 📊 تحلیلگر احساسات بازار مالی
    
    ### اخبار، گزارش‌ها و تحلیل‌های مالی را وارد کنید.
    
    💡 **کاربرد:** تحلیل اخبار بورس، گزارش‌های سودآوری، تحلیل‌های اقتصادی
    
    ---
    """)
    
    with gr.Tabs():
        
        # ========== تب ۱: تحلیل متن مالی ==========
        with gr.TabItem("📝 تحلیل خبر مالی"):
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        lines=6,
                        placeholder="""
📊 اخبار و تحلیل‌های مالی خود را وارد کنید...

مثال: 
"شرکت فولاد با افزایش ۲۰ درصدی سودآوری مواجه شد."
"قیمت دلار در بازار امروز با کاهش ۵ درصدی همراه بود."
                        """,
                        label="📝 متن خبر یا تحلیل مالی"
                    )
                    
                    with gr.Row():
                        analyze_btn = gr.Button("🔍 تحلیل", variant="primary")
                        clear_btn = gr.Button("🗑️ پاک کردن", variant="secondary")
                    
                    gr.Markdown("### 📚 مثال‌های بازار مالی")
                    gr.Examples(
                        examples=[
                            ["شرکت فولاد مبارکه با افزایش ۲۰ درصدی سودآوری در سه ماهه اخیر مواجه شد و چشم‌انداز روشنی دارد."],
                            ["قیمت نفت در بازارهای جهانی به شدت سقوط کرد و پیش‌بینی می‌شود روند نزولی ادامه یابد."],
                            ["بانک مرکزی نرخ بهره را کاهش داد که می‌تواند به رشد بازار سهام کمک کند."],
                            ["ارزش سهام شرکت خودروسازی به دلیل کاهش تقاضا با افت شدیدی مواجه شد."],
                            ["سرمایه‌گذاران خارجی با اعتماد به بازار ایران، حجم سرمایه‌گذاری خود را افزایش دادند."],
                            ["افزایش نرخ ارز و تورم، چشم‌انداز اقتصاد را مبهم کرده است."]
                        ],
                        inputs=text_input,
                        label="📊 برای تست کلیک کنید"
                    )
                
                with gr.Column(scale=1):
                    result_html = gr.HTML(
                        value='<div class="big-result-default big-result">⏳ منتظر ورودی...</div>'
                    )
                    
                    result_details = gr.JSON(
                        label="📋 جزییات تحلیل",
                        value={"نتیجه": "⏳ منتظر ورودی..."}
                    )
        
        # ========== تب ۲: آپلود فایل ==========
        with gr.TabItem("📂 آپلود اخبار مالی"):
            gr.Markdown("""
            ### 📂 آپلود فایل متنی برای تحلیل اخبار مالی
            
            **پشتیبانی از:** فایل‌های `.txt` با کدگذاری UTF-8
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(
                        label="📤 فایل خبری خود را آپلود کنید",
                        file_types=[".txt"],
                        file_count="single"
                    )
                    
                    analyze_file_btn = gr.Button("🔍 تحلیل اخبار", variant="primary")
                    
                    download_csv = gr.File(
                        label="📥 دانلود نتایج (CSV)",
                        interactive=False
                    )
                
                with gr.Column(scale=1):
                    file_status = gr.Markdown(
                        value="⏳ منتظر آپلود فایل...",
                        label="📊 وضعیت تحلیل"
                    )
            
            gr.Markdown("### 📋 نمونه نتایج")
            sample_output = gr.Dataframe(
                headers=["خط", "متن", "نتیجه", "امتیاز", "توصیه"],
                label="نمونه نتایج (۱۰ خط اول)",
                interactive=False
            )
    
    # ============================================
    # رویدادها
    # ============================================
    
    def process_text(text):
        if not text or text.strip() == "":
            return (
                '<div class="big-result-default big-result">⚠️ لطفاً متن را وارد کنید</div>',
                {"نتیجه": "⚠️ خطا", "درصد اطمینان": "لطفاً متن را وارد کنید"}
            )
        
        result = analyze_financial_text(text)
        
        label = result['نتیجه']
        if 'صعودی' in label:
            css = "big-result-positive"
        elif 'نزولی' in label:
            css = "big-result-negative"
        elif 'خنثی' in label:
            css = "big-result-neutral"
        else:
            css = "big-result-default"
        
        html = f"""
        <div class="{css} big-result">
            {label}<br>
            <span style="font-size: 28px; font-weight: 500;">
                اطمینان: {result['درصد اطمینان']}
            </span>
            <br>
            <span style="font-size: 26px; font-weight: 500;">
                {result['توصیه']}
            </span>
        </div>
        """
        
        return html, result
    
    analyze_btn.click(
        fn=process_text,
        inputs=text_input,
        outputs=[result_html, result_details]
    )
    
    text_input.change(
        fn=process_text,
        inputs=text_input,
        outputs=[result_html, result_details]
    )
    
    def clear_all():
        default_html = '<div class="big-result-default big-result">⏳ منتظر ورودی...</div>'
        default_json = {"نتیجه": "⏳ منتظر ورودی..."}
        return "", default_html, default_json
    
    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[text_input, result_html, result_details]
    )
    
    def process_file(file_obj):
        summary, csv_path, sample_df = analyze_financial_file(file_obj)
        if csv_path and sample_df is not None:
            return summary, csv_path, sample_df
        return summary, None, None
    
    analyze_file_btn.click(
        fn=process_file,
        inputs=file_input,
        outputs=[file_status, download_csv, sample_output]
    )
    
    # ============================================
    # فوتر
    # ============================================
    gr.Markdown("""
    ---
    ### 📈 تحلیلگر احساسات بازار مالی
    
    **✨ طراحی و توسعه با ❤️ توسط هلدینگ هوش مصنوعی سوفیا**
    
    🌐 [www.sofiaai.ir](https://www.sofiaai.ir)
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)