## toy flow demo

Haoxing Du, AY250 final project

This is a web application demo that uses normalizing flows to solve a toy problem.
The backend is served with Flask, and the frontend is written with React.

To run the app in development mode, run 
    
    pip install -r requirements.txt
    npm install
    npm start

In a different terminal, go into the `backend` directory, and run

    export FLASK_APP=server
    flask run

*Thanks to Peter Schmidt-Nielsen for teaching me React programming!*