from flask import Flask, render_template, request, session, redirect
import pickle
import random
import time
import json
import os

app = Flask(__name__)

# Set a secret key for the session
app.secret_key = os.urandom(24)

with open('/mnt/G-SSD/marco_mestrado/master_debias/pickle files/debiased_imgs_firefighter.pkl', 'rb') as f:
    debiased_imgs_firefighter = pickle.load(f)
with open('/mnt/G-SSD/marco_mestrado/master_debias/pickle files/original_imgs_firefighter.pkl', 'rb') as f:
    original_imgs_firefighter = pickle.load(f)

with open('/mnt/G-SSD/marco_mestrado/master_debias/pickle files/debiased_imgs_nurse.pkl', 'rb') as f:
    debiased_imgs_nurse = pickle.load(f)
with open('/mnt/G-SSD/marco_mestrado/master_debias/pickle files/original_imgs_nurse.pkl', 'rb') as f:
    original_imgs_nurse = pickle.load(f)

with open('/mnt/G-SSD/marco_mestrado/master_debias/pickle files/debiased_imgs_business_leader.pkl', 'rb') as f:
    debiased_imgs_business_leader = pickle.load(f)
with open('/mnt/G-SSD/marco_mestrado/master_debias/pickle files/original_imgs_business_leader.pkl', 'rb') as f:
    original_imgs_business_leader = pickle.load(f)

# load all available images
imgs_list = debiased_imgs_firefighter+original_imgs_firefighter+debiased_imgs_nurse+original_imgs_nurse+debiased_imgs_business_leader+original_imgs_business_leader
random.shuffle(imgs_list) # shuffles the order of images at randomF

@app.route('/')
def index():
        # Check if the user is logged in
    if 'user_id' in session:
        analyzed_imgs = []
        dir = '/mnt/G-SSD/marco_mestrado/master_debias/UI_label_images/labeled_images/'+session['user_id']+'/'
        for file in os.listdir(dir):
            if file.endswith('.pkl'):
                with open(dir+file, 'rb') as f:
                    analyzed_imgs.append(pickle.load(f))
        analyzed_imgs_path = []
        for i in range(len(analyzed_imgs)):
            for j in range(len(analyzed_imgs[i])):
                analyzed_imgs_path.append(analyzed_imgs[i][j]['img_path'])
        analyzed_imgs_path

        new_imgs = []
        for i in range(len(imgs_list)):
            if imgs_list[i]['img_path'] not in analyzed_imgs_path:
                new_imgs.append(imgs_list[i])
        
        if new_imgs == []:
            return render_template('imgs_done.html', error=None)
        else:
            # Fetch user-specific data or render the main page
            return render_template('index.html', image_data=new_imgs[0:30], user_id=session['user_id'])
    else:
        return redirect('/login')

@app.route('/submit', methods=['POST'])
def submit():
    updated_data = request.json  # Get updated data from the frontend
    if updated_data:
        with open('/mnt/G-SSD/marco_mestrado/master_debias/UI_label_images/labeled_images/'+updated_data[0]['user_name']+'/annotated_imgs_'+'_'+str(time.time())+'.pkl', 'wb') as f:
            pickle.dump(updated_data, f)
        return 'Success'


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        with open('/mnt/G-SSD/marco_mestrado/master_debias/UI_label_images/user_mapping.json') as f_in:
            userMap = json.load(f_in)
        # Check user credentials (example: hardcoded for demo)
        if username in userMap:
            session['user_id'] = username
            return redirect('/')
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html', error=None)

@app.route('/logout')
def logout():
    # Clear the user session data upon logout
    session.pop('user_id', None)
    return 'Logged out successfully'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004)