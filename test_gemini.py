import google.generativeai as genai


genai.configure(api_key="AIzaSyCkwdnrSZisXj7arazOy0MBoB6uZL0_pps")


model = genai.GenerativeModel("models/gemini-2.5-flash")

response = model.generate_content("Hello Gemini 2.5! Say hi in one short sentence.")
print(response.text)
