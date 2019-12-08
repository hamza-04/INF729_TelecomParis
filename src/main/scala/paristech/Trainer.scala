package paristech


import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.{HashingTF, IDF, OneHotEncoderEstimator, RegexTokenizer, StopWordsRemover, StringIndexer, Tokenizer}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}


/*******************************************************************************
  *
  *       TP 3
  *
  *       - lire le fichier sauvegarder précédemment
  *       - construire les Stages du pipeline, puis les assembler
  *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
  *       - Sauvegarder le pipeline entraîné
  *
  *       if problems with unimported modules => sbt plugins update
  *
  ********************************************************************************/

object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()

    val dftrain = spark.read.parquet("prepared_trainingset")

    // Stage 1 : récupérer les mots des textes
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")


    // Stage 2 : enlever les mots sans interets

    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filter_tokens")

    // Stage 3 : computer la partie TF

    val cvModel = new CountVectorizer()
      .setInputCol("filter_tokens")
      .setOutputCol("vectorized")

    val idf = new IDF().setInputCol("vectorized").setOutputCol("tfidf")

    //  Stage 5 : convertir country2 en quantités numériques
    val indexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setHandleInvalid("keep")

    //  Stage 6 : convertir currency2 en quantités numériques
    val indexer1 = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    // Stages 7 et 8: One-Hot encoder ces deux catégories
    val encoder = new OneHotEncoderEstimator()
      .setInputCols(Array("country_indexed", "currency_indexed"))
      .setOutputCols(Array("country_onehot", "currency_onehot"))

    // Stage 9 : assembler tous les features en un unique vecteur

    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf","days_campaign","hours_prepa","goal","country_onehot","currency_onehot"))
      .setOutputCol("features")


    // Stage 10 : créer/instancier le modèle de classification
      val lr = new LogisticRegression()
        .setElasticNetParam(0.0)
        .setFitIntercept(true)
        .setFeaturesCol("features")
        .setLabelCol("final_status")
        .setStandardization(true)
        .setPredictionCol("predictions")
        .setRawPredictionCol("raw_predictions")
        .setThresholds(Array(0.7, 0.3))
        .setTol(1.0e-6)
        .setMaxIter(50)

     // val lrModel = lr.fit(data)

    // pipeline , split , entrainement et test

    val pipeline = new Pipeline().setStages(Array(tokenizer,remover,cvModel,idf,indexer,indexer1,encoder,assembler,lr))

    val Array(training, test) = dftrain.randomSplit(Array(0.9, 0.1), seed = 12345)

    // Fit the pipeline to training documents.
    val model1 = pipeline.fit(training)

    val dfWithSimplePredictions=model1.transform(test)
    dfWithSimplePredictions.groupBy("final_status", "predictions").count.show(100)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")
    val f1_score = evaluator.evaluate(dfWithSimplePredictions)
    println("Test for metric f1 = " + f1_score)

    val paramGrid = new ParamGridBuilder()
      .addGrid(cvModel.minDF, Array(55.0, 75.0, 95.0))
      .addGrid(lr.regParam,Array(10e-8, 10e-6, 10e-4,10e-2))
      .build()

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    val cvModel0 = trainValidationSplit.fit(training)
    val dfWithSimplePredictionsO=cvModel0.transform(test).select("features", "final_status", "predictions")

    val f1_score0 = evaluator.evaluate(dfWithSimplePredictionsO)
    println("Test for metric f1 après utilisation de la grid = " + f1_score0)

    dfWithSimplePredictionsO.groupBy("final_status", "predictions").count.show(10)
    val finalTable:DataFrame=dfWithSimplePredictionsO.groupBy("final_status", "predictions").count()

    finalTable.write.parquet("sauvegarde_TP3.parquet")

  }
}
