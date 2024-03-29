###Clean version of chatgpt_summary_ed_final_220923.ipynb

##CYKW functions - PySpark
from pyspark.sql.functions import col
from pyspark.sql.functions import lower
from pyspark.sql.functions import isnan
from pyspark.sql.functions import lit, datediff, to_date
from pyspark.sql.functions import when, count
from pyspark.sql.functions import year, to_timestamp
from pyspark.sql.functions import log
from pyspark.sql.types import IntegerType


import pyspark.sql.functions as F

def print_if(*args):
  if display_print == 'Y':
    print(*args)
  else:
    print('CYKW: Output not printed; to print these outputs, change display_print = \'Y\'')
    pass

def shape(spark_df):
  if display_print == 'Y':
    print_if('(', spark_df.count(), ',', len(spark_df.columns), ')')
  else:
    pass
  
def value_counts(spark_df, column):
  if display_print == 'Y':
    print_if(spark_df.groupBy(column).count().orderBy('count', ascending = False).show(200, truncate = False))
    print_if('(Returns the first 200 results)')
  else:
    pass
  
def merge(spark_df1, spark_df2, left_on, right_on, how):
  #To prevent duplication 'on' columns, need to rename and then drop:
  spark_df2 = spark_df2.withColumnRenamed(right_on, 'right_on')
  merged = spark_df1.join(spark_df2, spark_df1[left_on] == spark_df2['right_on'], how).drop('right_on')
  return merged
  
def head(spark_df, n):
  if display_print == 'Y':
    print_if(spark_df.show(n, vertical = True, truncate = False))
  else:
    pass
  
def sample(spark_df, n):
  if display_print == 'Y':
    print_if(spark_df.sample(fraction=1.0).show(n, vertical = True, truncate = False))
  else:
    pass
  
def sample_df(spark_df, n):
    n_denominator = spark_df.count()
    return spark_df.sample(False, fraction=(n/denominator), seed = 42).limit(n)

  
def nunique(spark_df, column):
  print_if(spark_df.select(column).distinct().count())
  
def sort_values(spark_df, column1, column2, how):
  if how == 'descending':
    if column2 is None:
      sorted = spark_df.sort(col(column1).desc())
    if column2 is not None:
      sorted = spark_df.sort(col(column1).desc(), col(column2).desc())
  if how == 'ascending':
    if column2 is None:
      sorted = spark_df.sort(col(column1).asc())
    if column2 is not None:
      sorted = spark_df.sort(col(column1).asc(), col(column2).asc())
  return sorted

def drop_duplicates(spark_df, subset):
  print_if('drop_duplicates: keep = \'first\' ')
  dropped = spark_df.drop_duplicates(subset)
  return dropped

def count_null(spark_df, column):
  if display_print == 'Y':
    print_if(spark_df.filter((spark_df[column] == "")|spark_df[column].isNull()).count())
  else:
    pass
  
def count_notnull(spark_df, column):
  if display_print == 'Y':
    print_if(spark_df.filter(col(column).isNotNull()).count())
  else:
    pass
  
def return_notnull_df(spark_df, column):
  return spark_df.filter(col(column).isNotNull())

def return_isnull_df(spark_df, column):
  return spark_df.filter(col(column).isNull())

def register_table(spark_df, name):
  spark_df.registerTempTable(name)

def rename_columns(df, columns):
    if isinstance(columns, dict):
        return df.select(*[F.col(col_name).alias(columns.get(col_name, col_name)) for col_name in df.columns])
    else:
        raise ValueError("'columns' should be a dict, like {'old_name_1':'new_name_1', 'old_name_2':'new_name_2'}")
        
def isnull(df, columns):
  if display_print == 'Y':
    print_if(df.filter(col(columns).isNull()).count())
  else:
    pass


##Set whether to print values (takes up time)
display_print = 'Y'

##Display non-truncated text
spark.conf.set("spark.sql.repl.eagerEval.maxToStringFields", 1000)

###############################

####Generate initial master ED dataframe

###Load ed_table
path = 'setpath'

pd.set_option('display.max_rows', 200)
ed_table = spark.read.options(header='true', inferschema='true').parquet(path + 'ed_table/')
print(ed_table.count())
print(nunique(ed_table, 'visit_occurrence_id'))

##Select only relevant columns:
columns_to_select = ['person_id', 'agekey', 'admission_visit_occurrence_id', 'visit_occurrence_id', 'arrival_datetime', 'departure_datetime', 'disposition_datetime', 'dischargedisposition', 'acuitylevel']
ed_table = ed_table.select([col(column) for column in columns_to_select])
print(ed_table.columns)

### Add in age, sex demographics
patient_key_table = spark.read.options(header='true', inferschema='true').parquet(path + 'patient_key_table/')
print(patient_key_table.columns)

#Add in sex
ed_table = merge(ed_table, patient_key_table.select(col('person_id'), col('sex')), left_on = 'person_id', right_on = 'person_id', how = 'left')

#Merge agekey with keys in duration_table to retrieve age
duration_table = spark.read.options(header='true', inferschema='true').parquet(path + 'duration_table/')

ed_table = merge(ed_table, duration_table.select(col('durationkey'), col('days')), left_on = 'agekey', right_on = 'durationkey', how = 'left')
ed_table = ed_table.withColumn('edvisit_age', col('days') / 365.25)


###Import notes
note_metadata = spark.read.options(header='true', inferschema='true').parquet(path + 'note_metadata')
print(note_metadata.count())

note_text = spark.read.options(header='true', inferschema='true').parquet(path + 'note_text')
print(note_text.count())

#Select relevant columns
notes = note_metadata.select(col('person_id'), col('deid_note_key'), col('visit_occurrence_id'), col('note_type'), col('encounter_type'), col('enc_dept_name'), col('enc_dept_specialty'),col('auth_prov_type'), col('prov_specialty'), col('deid_service_date'))
#Add note_text
notes = merge(notes, note_text, left_on = 'deid_note_key', right_on = 'deid_note_key', how = 'left')

notes = rename_columns(notes, {'person_id':'person_id_notes'})


### Retrieve ed_table notes
#Further reduce columns in ed_table
print(ed_table.columns)
columns_to_select2 = ['person_id', 'admission_visit_occurrence_id', 'visit_occurrence_id', 
                      'arrival_datetime', 'admissiondecision_datetime', 'departure_datetime', 'disposition_datetime', 
                      'arrivalmethod', 'dischargedisposition',
                      'eddisposition', 'acuitylevel', 'sex', 'edvisit_age']
ed_table_selected = ed_table.select([col(column) for column in columns_to_select2])
ed_notes = merge(ed_table_selected, notes, left_on = 'visit_occurrence_id', right_on = 'visit_occurrence_id', how = 'inner')

### Filter adults only
nunique(return_isnull_df(ed_notes, 'edvisit_age'), 'visit_occurrence_id')
ed_notes_adults = ed_notes.filter(col('edvisit_age') >= 18)
nunique(ed_notes_adults, 'visit_occurrence_id')
print(ed_notes_adults.count())


### Filter to note_type = 'ED Provider Notes'
ed_notes_adults_edprovider = ed_notes_adults.filter(col('note_type') == 'ED Provider Notes')
nunique(ed_notes_adults_edprovider, 'visit_occurrence_id')
print(ed_notes_adults_edprovider.count())


### Create ed_notes_edprovider_adults
#Filter by prov_specialty (including only Emergency Medicine and UCSF)
ed_notes_edprovider_adults_filtered = ed_notes_adults_edprovider.filter((col('prov_specialty') == 'Emergency Medicine')|(col('prov_specialty') == 'UCSF'))
nunique(ed_notes_edprovider_adults_filtered, 'visit_occurrence_id')
print(ed_notes_edprovider_adults_filtered.count())


#### Save PySpark Dataframe
ed_notes_edprovider_adults_filtered.write.mode("overwrite").option("mergeSchema", "true").parquet(path + "chatgpt_summary_ed_ed_notes_edprovider_adults_filtered_master_220923.parquet")