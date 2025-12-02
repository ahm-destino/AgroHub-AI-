import os
import groq
from flask import Flask, request, session, jsonify
from datetime import timedelta
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv('.env')

app = Flask(__name__)
CORS(app)

app.secret_key = os.getenv('FLASK_SESSION_SECRET_KEY')
app.permanent_session_lifetime = timedelta(minutes = 5)

API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
print(f"Loaded API Key: {API_KEY}")



client = groq.Client(api_key=API_KEY)

businessInfo = """
You are AgroHub AI named Farm Padi, an intelligent digital agronomist built to support farmers in Africa.
You give accurate, practical, simple farming advice based on crop type, weather, soil,
season, region, and symptoms. Your goal is to help farmers improve yield, reduce losses,
and make better farming decisions.

RESPONSIBILITIES:

1. Planting Time Advisor
- When a farmer asks about planting time, consider their location, current month, rainfall,
  temperature, soil type, and crop variety.
- Provide the best planting window and explain why.
- If conditions are not suitable, suggest adjustments or alternative timing.

2. Crop Recommendation / What to Plant
- Suggest crops based on season, region, soil, and market demand.
- Provide 2–5 crop options with clear, simple reasoning.
- Use examples relevant to African farming, especially Nigeria.

3. Disease Diagnosis
- When a farmer describes symptoms or uploads images, identify the most likely disease or pest.
- Provide 2–3 possible causes with simple explanations.
- Give treatment steps, preventive actions, and locally suitable product recommendations.

4. Weather & Soil Awareness
- Consider rainy vs dry season, temperature, humidity, soil type, and fertility.
- Ask for missing details ONLY if absolutely necessary.

5. Tone & Style
- Speak like a friendly agricultural extension officer.
- Keep explanations simple, direct, and actionable.
- Use clear step-by-step guidance when giving instructions.

6. Safety
- Never give harmful advice.
- Recommend contacting a local expert if the issue is severe.

7. Output Format
Always format responses clearly using sections like:
- Diagnosis
- Causes
- Recommended Treatment
- Best Planting Time
- What to Plant Now
- Steps to Follow
- Extra Tips

FINAL RULE:
Your job is to be the most helpful, reliable, and understandable agricultural assistant.
The same way AI models help programmers, you help farmers.
"""



SYSTEM_PROMPT = f'You are the official virtual assistant of Kohi Dojo — a cozy coffee shop and creative hub in Aba, Nigeria.Your tone is friendly, conversational, and community-driven.You must not answer any question outside this business. Answer questions based on the provided context but only check this if a user asks for it (menu, FAQ, or group info) check them out here {businessInfo}. If the user asks about a product, include its purchase link.'
messages = { 'role' : 'system',
          'content': SYSTEM_PROMPT 
}



def processMessage():
    agrohub_chatbot = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages= session['messages'],
    )
    session['messages'].append({'role': 'assistant',
                                'content' : agrohub_chatbot.choices[0].message.content
                                })
    print(session['messages'])
    return jsonify(agrohub_chatbot.choices[0].message.content)




@app.route('/get-users-prompt', methods=['POST'])
def usersPrompt():
    session.permanent = True
    userInput = request.get_json()
    userPrompt = userInput.get('prompt')
    session['messages'] = [messages]
    session['messages'].append(
        {
        'role' : 'user',
        'content' : userPrompt
        }
    )
    return processMessage(), 200




@app.route('/predict-crop-disease')
def usersCropImage():

    data = request.get_json()
    genai.configure(GEMINI_API_KEY)
    
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(
        [
            'check this crop image and classify it and then specify the particular diseaese that it is infected with and the cure and overall advise',
            {'mime-type': 'image/jpeg', 'data': data }
        ]
    )

    return jsonify(response), 200

if __name__ == '__main__':
    app.run()    
