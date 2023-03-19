from flask import Flask, render_template

app = Flask(__name__)

# Define the route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/audio')
def audio():
    return render_template('audio.html')
@app.route('/breath')
def breath():
    return render_template('breath.html')
@app.route('/hand')
def hand():
    return render_template('hand.html')

if __name__ == '__main__':
    app.run(port=5000, debug=True)

