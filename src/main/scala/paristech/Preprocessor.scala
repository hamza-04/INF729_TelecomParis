package paristech


import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.functions.hour
import org.apache.spark.sql.functions.from_unixtime

object Preprocessor {

  def main(args: Array[String]): Unit = {

    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      * root
      * |-- project_id: string (nullable = true)
      * |-- name: string (nullable = true)
      * |-- desc: string (nullable = true)
      * |-- goal: integer (nullable = true)
      * |-- keywords: string (nullable = true)
      * |-- disable_communication: string (nullable = true)
      * |-- country: string (nullable = true)
      * |-- currency: string (nullable = true)
      * |-- deadline: integer (nullable = true)
      * |-- state_changed_at: string (nullable = true)
      * |-- created_at: string (nullable = true)
      * |-- launched_at: string (nullable = true)
      * |-- backers_count: integer (nullable = true)
      * |-- final_status: integer (nullable = true)
      * Ajoutez une colonne days_campaign qui représente la durée de la campagne en jours (le nombre de jours entre launched_at et deadline).
      ********************************************************************************/

    // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
    // On vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Preprocessor")
      .getOrCreate()


    val df: DataFrame = spark
      .read
      .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .csv("/Users/hamzaamri/Desktop/Spark project/spark_project_2/data/train_clean.csv")//.icloud")

    import spark.implicits._

    val dfCasted: DataFrame = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline", $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))


    val df2: DataFrame = dfCasted.drop("disable_communication")


    // Les fuites du future

    val dfNoFutur: DataFrame = df2.drop("backers_count", "state_changed_at")

    def cleanCountry(country: String, currency: String): String = {
      if (country == "False")
        currency
      else
        country
    }

    def cleanCurrency(currency: String): String = {
      if (currency != null && currency.length != 3)
        null
      else
        currency
    }

    val cleanCountryUdf = udf(cleanCountry _)
    val cleanCurrencyUdf = udf(cleanCurrency _)

    val dfCountry: DataFrame = dfNoFutur
      .withColumn("country2", cleanCountryUdf($"country", $"currency"))
      .withColumn("currency2", cleanCurrencyUdf($"currency"))
      .drop("country", "currency")

    import org.apache.spark.sql.functions._

    def Times_hour(launched_at: Int, created_at: Int): Float = {

      (launched_at-created_at)/3600
    }
    def Times_day(launched_at: Int, created_at: Int): Float = {

      (launched_at-created_at)/86400
    }

    val timefunctionUdf_h = udf(Times_hour _)
    val timefunctionUdf_d = udf(Times_day _)
    val dfprepa: DataFrame = dfCountry
        .withColumn("hours_prepa", round(timefunctionUdf_h($"launched_at", $"created_at"),5))
        .withColumn("days_campaign", round(timefunctionUdf_d($"launched_at", $"created_at"),3))

      dfprepa
        .drop("launched_at", "created_at","deadline")

    // lowercase

    def Upper_words(s: String): String = {
      if(s!=null)
        s.toLowerCase
      else
        "UNKNOWN"
    }
    val textfunctionUdf = udf(Upper_words _)

    // fonction de concatenisation
    def Concat_words(s: String,a: String,b: String): String = {s+" "+a+" "+b }
    val concatfunctionUdf = udf(Concat_words _)

    // lowercase des colonnes et creer une colonne text qui concatenise le reste
    val dftext:DataFrame=dfprepa
      .withColumn("name", textfunctionUdf($"name"))
      .withColumn("desc", textfunctionUdf($"desc"))
      .withColumn("keywords", textfunctionUdf($"keywords"))
      .withColumn("text", concatfunctionUdf($"name",$"desc",$"keywords"))

    val dfreplace:DataFrame=dftext
        .withColumn("days_campaign", when($"days_campaign".isNull , -1.0).otherwise($"days_campaign"))
        .withColumn("hours_prepa", when($"hours_prepa".isNull , -1.0).otherwise($"hours_prepa"))
        .withColumn("goal", when($"goal".isNull , -1.0).otherwise($"goal"))
        .withColumn("deadline", when($"deadline".isNull , -1.0).otherwise($"deadline"))
        .withColumn("country2", when($"country2".isNull , "UNKNOWN").otherwise($"country2"))
        .withColumn("currency2", when($"currency2".isNull , "UNKNOWN").otherwise($"currency2"))

    val df_final:DataFrame=dfreplace

    // sauvegarde du fichier en parquet :
    dfreplace.write.parquet("sauvegarde_TP2.parquet")

    println("\n")
    println(s"Nombre de lignes : ${df.count}")
    println("\n")
    println("\n")

  }
}
