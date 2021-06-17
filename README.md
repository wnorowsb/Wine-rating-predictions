# Wine scoring prediction
This project is a fully implemented ML pipeline. It's functionality is to:
- download dataset
- split data into train and testing parts
- clean datasets independently from redundant columns and handle missing values
- extract features from some text values, perform encoding on other
- train Ridge Regression to predict points graded to a wine
- evaluate trained model, output metrics and plots based on it's performance

Each of those steps was assigned to separate Luigi task. You can learn more on how Luigi works [here](https://luigi.readthedocs.io/en/stable/).


You will need docker an docker-compose to run this repository: 

* [How to install docker](https://docs.docker.com/engine/installation/)
* [How to install docker-compose](https://docs.docker.com/compose/install/)

After you get those tools, there is only two steps necessary to trigger the whole pipeline:

1. Build docker images which arent specified in ```docker-compose.yml```

    ```./build-task-images.sh 0.1```
2. Run main container and create whole network by ```docker-compose```

    ```docker-compose up orchestrator```

After that, you should get a lot of logs from the ```orchestrator``` container. The whole process can take a few minutes. While finished, the last log you should see is: 

``` orchestrator_1    | This progress looks :) because there were no failed tasks or missing dependencies ```

Now you can get the trained model ```pickle``` file, performance metrics and the plots from the ``` data_root/output ``` folder. 

## Please check `notebooks/` if you want to have a clear view on model creation and skip the engineering part.
