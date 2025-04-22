import pyttsx3
import speech_recognition as sr
import datetime
import wikipedia
import webbrowser
import os
import smtplib
import pyjokes
import time
import psutil
import requests
import json
from plyer import notification
import subprocess
import numpy as np
import librosa
import joblib
import sounddevice as sd

# Voice authentication config
SAMPLE_RATE = 22050
DURATION = 3
MODEL_PATH = "voice_model.pkl"

# Load voice authentication model
voice_model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# Initialize text-to-speech engine
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

def speak(text):
    print(f"Jarvis: {text}")
    engine.say(text)
    engine.runAndWait()

def wishMe():
    hour = datetime.datetime.now().hour
    if hour < 12:
        speak("Good Morning!")
    elif 12 <= hour < 18:
        speak("Good Afternoon!")
    else:
        speak("Good Evening!")
    speak("I am Jarvis, your AI assistant. Say 'ok jarvis' to activate me.")

def takeCommand(timeout=5):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source)
        try:
            audio = r.listen(source, timeout=timeout)
            query = r.recognize_google(audio, language='en-in')
            print(f"User said: {query}")
            return query.lower()
        except Exception as e:
            print("Error recognizing speech:", e)
            return None

def record_for_auth():
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def extract_features(audio, sample_rate=SAMPLE_RATE):
    try:
        target_len = sample_rate * DURATION
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]

        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)

        combined = np.hstack([ 
            np.mean(mfcc.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(contrast.T, axis=0)
        ])
        return combined.reshape(1, -1)

    except Exception as e:
        print(f"Feature extraction failed during authentication: {e}")
        return None

def sendEmail(to, content):
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login('vermadhawal86@gmail.com', 'Dv862004')
        server.sendmail('vermadhawal86@gmail.com', to, content)
        server.close()
        speak("Email sent successfully!")
    except Exception:
        speak("Sorry, I couldn't send the email.")

def get_weather(city="Delhi"):
    API_KEY = "77ce5bf2525bc46df87f2c01d0c389e9"
    BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

    try:
        params = {
            'q': city,
            'appid': API_KEY,
            'units': 'metric'
        }
        response = requests.get(BASE_URL, params=params)
        data = response.json()

        if data["cod"] != 200:
            speak(f"Sorry, I couldn't find weather data for {city}.")
            return

        weather = data['weather'][0]['description']
        temp = data['main']['temp']
        feels_like = data['main']['feels_like']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']

        report = (
            f"The current weather in {city} is {weather}. "
            f"The temperature is {temp}°C, feels like {feels_like}°C. "
            f"Humidity is at {humidity}% and wind speed is {wind_speed} meters per second."
        )

        speak(report)

    except Exception as e:
        print(f"Weather API error: {e}")
        speak("I ran into an issue while fetching the weather.")

def get_news():
    API_KEY = "4b4f0efc7fed412fa23c2e3a6c6832ea"
    BASE_URL = "https://newsapi.org/v2/top-headlines"

    try:
        params = {
            'country': 'us',  # You can change this to your desired country (e.g., 'in' for India)
            'apiKey': API_KEY
        }
        response = requests.get(BASE_URL, params=params)
        data = response.json()

        if data["status"] != "ok":
            speak("Sorry, I couldn't fetch the news right now.")
            return

        articles = data["articles"]
        if not articles:
            speak("No news articles found.")
            return

        speak("Here are the top news headlines:")
        for article in articles[:5]:  # Limiting to top 5 articles
            title = article['title']
            description = article['description']
            speak(f"Title: {title}")
            print(f"Title: {title}")
            if description:
                speak(f"Description: {description}")
                print(f"Description: {description}")
            speak("------")

    except Exception as e:
        print(f"News API error: {e}")
        speak("I ran into an issue while fetching the news.")

def battery_status():
    battery = psutil.sensors_battery()
    speak(f"Battery level is at {battery.percent} percent.")

def take_note():
    speak("What should I write?")
    note = takeCommand()
    if note:
        with open("notes.txt", "a") as file:
            file.write(f"{datetime.datetime.now()} - {note}\n")
        speak("Note saved successfully.")

def reminder():
    speak("What should I remind you about?")
    task = takeCommand()
    if not task:
        return
    speak("In how many seconds should I remind you?")
    try:
        seconds = int(takeCommand())
        time.sleep(seconds)
        notification.notify(title="Reminder", message=task, timeout=5)
        speak(f"Reminder: {task}")
    except ValueError:
        speak("Invalid time input. Reminder cancelled.")

def open_application(app_name):
    apps = {
        "notepad": "notepad",
        "calculator": "calc",
        "command prompt": "cmd",
        "valorant": r"C:\\Riot Games\\Riot Client\\RiotClientServices.exe"
    }
    if app_name in apps:
        try:
            subprocess.Popen(apps[app_name], shell=True)
            speak(f"Opening {app_name}")
        except Exception as e:
            speak(f"Couldn't open {app_name}. Error: {e}")
    else:
        speak("Application not found.")

def shutdown_system():
    speak("Shutting down the system")
    os.system("shutdown /s /t 5")

def restart_system():
    speak("Restarting the system")
    os.system("shutdown /r /t 5")

def passive_authenticate_and_activate():
    print("Waiting for wake word: 'ok jarvis'...")
    while True:
        query = takeCommand(timeout=5)
        if query and "ok jarvis" in query:
            try:
                audio_data = record_for_auth()
                features = extract_features(audio_data)

                if voice_model and hasattr(voice_model, 'predict_proba'):
                    prob = voice_model.predict_proba(features)[0][1]
                    print(f"Authentication probability: {prob:.2f}")
                    if prob > 0.81:
                        hour = datetime.datetime.now().hour
                        greet = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
                        speak(f"{greet}! Access granted.")
                        return
                    else:
                        speak("Voice not similar enough. Access denied.")
                else:
                    speak("Authentication model is missing or incompatible.")
            except Exception as e:
                print("Authentication error:", e)
                speak("Authentication failed. Please try again.")

def active_mode():
    timeout = 60
    start_time = time.time()
    while time.time() - start_time < timeout:
        query = takeCommand(timeout=5)
        if not query:
            continue
        start_time = time.time()

        if 'wikipedia' in query:
            speak('Searching Wikipedia...')
            query = query.replace("wikipedia", "")
            try:
                results = wikipedia.summary(query, sentences=2)
                speak("According to Wikipedia")
                print(results)
                speak(results)
            except:
                speak("Sorry, couldn't fetch information from Wikipedia.")

        elif 'open youtube' in query:
            webbrowser.open("https://youtube.com")

        elif 'open google' in query:
            webbrowser.open("https://google.com")

        elif 'open stack overflow' in query:
            webbrowser.open("https://stackoverflow.com")

        elif 'play music' in query:
            music_path = r"C:\\Users\\dhawa\\Music\\Perfect-(Mr-Jat.in).mp3"
            os.startfile(music_path) if os.path.exists(music_path) else speak("Music file not found.")

        elif 'the time' in query:
            speak(datetime.datetime.now().strftime("%H:%M:%S"))

        elif 'open' in query:
            app_name = query.replace('open ', '')
            open_application(app_name)

        elif 'shutdown' in query:
            shutdown_system()

        elif 'restart' in query:
            restart_system()

        elif 'joke' in query:
            speak(pyjokes.get_joke())

        elif 'weather in' in query:
            city = query.split("weather in")[-1].strip()
            get_weather(city)

        elif 'news' in query:
            get_news()

        elif 'battery' in query:
            battery_status()

        elif 'take note' in query:
            take_note()

        elif 'reminder' in query:
            reminder()

        elif 'email to dhawal' in query:
            speak("What should I say?")
            content = takeCommand()
            if content:
                sendEmail("dhawalverma86@gmail.com", content)

        elif 'play valorant' in query or 'play valo' in query:
            open_application('valorant')

        elif 'exit' in query or 'quit' in query:
            speak("Goodbye! Have a great day.")
            exit()

    speak("Going back to sleep. Say 'ok Jarvis' to wake me again.")

if __name__ == "__main__":
    wishMe()
    while True:
        passive_authenticate_and_activate()
        active_mode()
