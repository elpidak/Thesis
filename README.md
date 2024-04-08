Instructions in order to run the project:

1. Download Carla version 0.9.8

2. Open the Simulator

3. Create a virtual environment and install the libraries from requirements.txt

4. If you want to run the already trained agent run the command:
    
    python continuous_driver.py --train True --town Town02 --load-checkpoint True 

5. If you want to train an agent from start run the command:
    
    python continuous_driver.py --train True --town Town02

    you can change the Town, but keep in mind that the vae is pretrained in Towns 2 and 7.

6. If you want to test the agent in Town2 run the command:

    python continuous_driver.py --town Town02


