#!/usr/bin/python  
# 
# 
# # ## Initialisation
# - Import libraries and functions
# - Create SparkSession using config options to support reading from S3 bucket
# - Set useful output formats and options


from itertools import chain
import pandas as pd

from pyspark.sql import SparkSession
import pyspark.sql.functions as f
import time
import sys

from pyspark.mllib.stat import Statistics


#jars = '/home/ubuntu/hadoop-2.7.3/share/hadoop/tools/lib/aws-java-sdk-1.7.4.jar:' \
#       '/home/ubuntu/hadoop-2.7.3/share/hadoop/tools/lib/hadoop-aws-2.7.3.jar'
    
#spark = SparkSession.builder.master("local").appName("USHealth") \
#                    .config('spark.hadoop.fs.s3a.impl', 'org.apache.hadoop.fs.s3a.S3AFileSystem') \
#                    .config('spark.driver.extraClassPath', jars) \
#                    .config('spark.hadoop.fs.s3a.access.key', 'xxxxxxxxxxxxxxxx') \
#                    .config('spark.hadoop.fs.s3a.secret.key', 'xxxxxxxxxxxxxxxxxxxxxxxxxxx') \
#                    .getOrCreate()
startTime = time.time()
if len(sys.argv) != 3:
    quit()

NCities = int(sys.argv[1])
NSlaves = int(sys.argv[2])

s3InputPath = "s3a://cca.project.bucket1/input/"
s3OutputPath = "s3a://cca.project.bucket1/output/"

latlonFilename = s3InputPath + "new_{}_lat_lon.csv".format(NCities)
citiesFilename = s3InputPath + "new_{}_cities.csv".format(NCities)
healthFilename = s3InputPath + "new_{}_health_ineq_all_online_tables.csv".format(NCities)
lmaFilename = s3InputPath + "new_{}_1990LMAascii.csv".format(NCities)
outputFilename = s3OutputPath + "output_{}_{}.csv".format(NCities, NSlaves)

if NSlaves == 0:
    spark = SparkSession.builder.master("local").appName("USHealth").getOrCreate()
else:
    spark = SparkSession.builder.master("yarn-client").appName("USHealth").getOrCreate()

#spark = SparkSession.builder.master("local").appName("USHealth").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# set Jupyter to display ALL output from a cell (not just last output)

# set pandas and numpy options to make print format nicer
pd.set_option('display.width',110); pd.set_option('display.max_columns',100)
pd.set_option('display.max_colwidth', 200); pd.set_option('display.max_rows', 500)

print("Start %d Cities Cluster Test, NSlaves = %d" % (NCities, NSlaves), flush=True)
print("Execution time #0 %f" % (time.time() - startTime),flush=True)

# ## Read two mapping files into dataframes
# - Read files from Amazon S3 bucket into Spark dataframes
# - Format columns as required to enable joins to dataset below

# read and process city FIPS to county FIPS mapping file
city_to_fips = spark.read.format("org.apache.spark.csv").option("header","true") \
                    .csv(latlonFilename)
#                          .csv("/home/ubuntu/project/data/uscitiesLatLongFIPS.csv")

city_to_fips = city_to_fips.withColumn("county_FIPS", f.lpad(city_to_fips['county_FIPS'],5,"0"))
city_to_fips = city_to_fips.drop("city","zip","id","source","population")
city_to_fips = city_to_fips.withColumn("city_ascii", f.regexp_replace('city_ascii', 'Saint', 'St.'))
city_to_fips = city_to_fips.withColumnRenamed("city_ascii","CityName") \
                           .withColumnRenamed("state_name","StateDesc") \
                           .withColumnRenamed("county_FIPS","FIPS")

print((city_to_fips.count(), len(city_to_fips.columns)))
city_to_fips.limit(5).toPandas()


# read and process commuting zone to county FIPS mappingfile
cz_to_fips = spark.read.format("org.apache.spark.csv").option("header","true").option("delimiter", "\t") \
                  .csv(lmaFilename)
#                        .csv("/home/ubuntu/project/data/1990LMAascii.csv")
    
cz_to_fips = cz_to_fips.filter(cz_to_fips.FIPS !="None")
cz_to_fips = cz_to_fips.withColumn("stateabbrv", cz_to_fips["County Name"].substr(-2,99))
cz_to_fips = cz_to_fips.withColumnRenamed("LMA/CZ","cz")
cz_to_fips = cz_to_fips.withColumn("cz", cz_to_fips["cz"].cast("Integer"))

print((cz_to_fips.count(), len(cz_to_fips.columns)))
cz_to_fips.limit(5).toPandas()


# ## Read Life Expectancy HDFS file into dataframe and process
# - Read file from S3 bucket into Spark dataframe and drop unneeeded columns
# - Join to cz_fips dataframe to get county FIPS code (this creates lots more rows for life expectancy by county)


life = spark.read.format("org.apache.spark.csv").option("header","true").option("comment",",") \
            .csv(healthFilename)
#                  .csv("/home/ubuntu/project/data/health_ineq_all_online_tables.csv")
print((life.count(), len(life.columns)))

life = life.drop("rownum","czname","pop2000","stateabbrv","fips", "_c7","_c16","_c25","_c34","_c43","_c48")                 
life = life.join(cz_to_fips, ["cz"], "left")

print((life.count(), len(life.columns)))
# life.limit(2).toPandas()


# ## Read Health HDFS file into dataframe and process
# 
# - Read file from S3 bucket into Spark dataframe
# - Set CityName to "Average" for United States so we have a National row
# - Filter out unneededed rows
# - Calculate health score for each health measure (row) by multiplying by weights
# - Format fields as required

health_R = spark.read.format("org.apache.spark.csv").option("header","true") \
                .csv(citiesFilename)
#                      .csv("/home/ubuntu/data/project/500_Cities.csv")
print((health_R.count(), len(health_R.columns)))
health_R.limit(5).toPandas()

health_R = health_R.withColumn("CityName", f.when(health_R["StateDesc"] == "United States", "Average")                                             .otherwise(health_R["CityName"]))

health_R = health_R.filter((health_R['Data_Value_Type']=='Crude prevalence') & 
                           (health_R['GeographicLevel'].isin(['City','US']) ))
print((health_R.count(), len(health_R.columns)))

health_R = health_R.select('StateDesc','CityName','Data_Value','PopulationCount','CityFIPS','Short_Question_Text')

weights = {'Health Insurance':        6,
           'Arthritis':               3,
           'Binge Drinking':          6,
           'High Blood Pressure':     7,
           'Taking BP Medication':    3,
           'Cancer (except skin)':   10,
           'Current Asthma':          4,
           'Coronary Heart Disease': 10,
           'Annual Checkup':         -3,
           'Cholesterol Screening':  -3,
           'Colorectal Cancer Screening': -2,
           'COPD'                       : 10,
           'Core preventive services for older men':   -3,
           'Core preventive services for older women': -3,
           'Current Smoking':         7,
           'Dental Visit':           -1,
           'Diabetes':                8,
           'High Cholesterol':        6,
           'Chronic Kidney Disease':  8,
           'Physical Inactivity':     3,
           'Mammography':            -2,
           'Mental Health':           6,
           'Obesity':                 7,
           'Pap Smear Test':         -2,
           'Sleep < 7 hours':         2,
           'Physical Health':         5,
           'Stroke':                  9,
           'Teeth Loss':              1,
          }


print("Execution time #1 %f" % (time.time() - startTime),flush=True)

mapping_expr = f.create_map([f.lit(x) for x in chain(*weights.items())])

health_R = health_R.withColumn("Weight", mapping_expr.getItem(f.col("Short_Question_Text")))
health_R = health_R.withColumn("Score", health_R.Data_Value * health_R.Weight)
health_R = health_R.withColumn("Data_Value", health_R["Data_Value"].cast("Double"))

health_R.printSchema()
health_R.limit(5).toPandas()

print("Execution time #2 %f" % (time.time() - startTime),flush=True)


# ## Build final total dataframe 
# - Build health1 dataframe with health data values all pivoted into 1 row for each city
# - Build health2 dataframe of distinct state, city, population and FIPS
# - Join the above two dataframes on state and city to get final health dataframe
# - Join this to the city_to_fips dataframe to add the county FIPS code 
# - Finally create the total dataframe as a join of health and life dataframes


health1 = health_R.groupBy("StateDesc", "CityName").pivot("Short_Question_Text").sum("Data_Value")
health2 = health_R.groupBy('StateDesc','CityName','PopulationCount','CityFIPS').sum("Score") \
                  .withColumnRenamed("sum(Score)","Raw Score")

# normalise health score to range 0 to 100
min_score, max_score = health2.agg(f.min("Raw Score"), f.max("Raw Score")).take(1)[0]
health2 = health2.withColumn("Score", 
                  (100*(1-(health2["Raw Score"] - min_score) / (max_score-min_score))).cast("Integer") )

# join back to previous dataframe to pickup info list in the groupby
health = health1.join(health2, ["StateDesc", "CityName"], "left")
print("Execution time #3 %f" % (time.time() - startTime),flush=True)

# join to the city_to_fips dataframe to get county FIPS code
health = health.join(city_to_fips, ["StateDesc","CityName"], "left")
# print((health.count(), len(health.columns)))
# health.limit(5).toPandas()

# build the total dataframe from health and life
total = health.join(life, ['FIPS'], 'left')
print((total.count(), len(total.columns)))
total.limit(5).toPandas()


print("\nCheck for any nulls - should be empty or just United States Average")
total.filter(total.count_q4_M.isNull()).toPandas()

print("Execution time #4 %f" % (time.time() - startTime),flush=True)


# create total summary dataframe
total.write.option("header", "true").csv(outputFilename)
print("Execution time #5 %f" % (time.time() - startTime),flush=True)


# tell spark to stop
spark.stop()

