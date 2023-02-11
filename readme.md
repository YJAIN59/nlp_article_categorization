# NLTK data
nltk.download('stopwords', download_dir=os.curdir)
nltk.download('punkt', download_dir=os.curdir)


# About this repo:
**'main'** branch has all the EDA, training, model testing code

**'app_only_branch'** is lightweight branch with Flask app. It ingests trained model created while training.

# Steps to Deploy the app:
0. Fork this repository into your Github account
1. Register to railway.app
2. Create a new project, then create new service
3. Import this repo **"app_only_branch"** 
4. Set the variable **"host" = "0.0.0.0", "PORT" = 5000**
5. Deployment will start.
6. Generate DNS from 'Settings' and open the url(dns) in new tab. In few minutes, deployment will complete.
