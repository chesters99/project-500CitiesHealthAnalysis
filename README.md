# US 500 Cities Health and Life Expectancy Analysis

This notebook uses Spark (via PySpark) to read HDFS files from an AWS S3 bucket, and to join them and perform some basic analysis (correlations). Pandas is used to improve output formattting.

### Abstract
Reducing health inequality has been declared an important item on the public health agenda by the World Health Organization. We have studied health inequality across US cities and states by analyzing data on health services coverage and the prevalence of chronic diseases using a cloud architecture. We measured the performance of the cloud platform and investigated how it scales with the expected future growth of the data set using synthetic data sets.

Our analysis revealed considerable differences between cities and states and offers insights into possible causal relationships. A ranking of cities based on a composite health score was created as a basis for guiding health system policy.
The cloud platform scaled with increasing data size and was shown to deliver satisfactory performance with a single-digit number of compute nodes for realistic real-world data size.

### Technical Notes
To login to AWS instance
ssh -i ./ccaprojectmoran.pem -L 8157:127.0.0.1:8888 ubuntu@ec2-54-159-29-234.compute-1.amazonaws.com

Password for jupyter notebook "ccarocks"
Home directory for jupyter is /home/ubuntu/project, so all notebooks and data are there (for now)

To access jupyter http://127.0.0.1:8157 in your own machine

Jupyter starts at system boot but in case of problems "sudo systemctl restart jupyter.service”. The "source activate python3" that is required is now in .profile so is done at login time

