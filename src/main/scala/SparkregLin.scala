import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.{SparkConf, SparkContext}

object SparkregLin {

  def main(args: Array[String]): Unit = {

    val conf= new SparkConf().setMaster("local[*]")
      .setAppName("SparkregressionLin")
    val sc= new SparkContext(conf)

    val points = Array(
      LabeledPoint(1620000,Vectors.dense(2100)),
      LabeledPoint(1690000,Vectors.dense(2300)),
      LabeledPoint(1400000,Vectors.dense(2046)),
      LabeledPoint(2000000,Vectors.dense(4314)),
      LabeledPoint(1060000,Vectors.dense(1244)),
      LabeledPoint(3830000,Vectors.dense(4608)),
      LabeledPoint(1230000,Vectors.dense(2173)),
      LabeledPoint(2400000,Vectors.dense(2750)),
      LabeledPoint(3380000,Vectors.dense(4010)),
      LabeledPoint(1480000,Vectors.dense(1959))
    )

    val pricesRDD = sc.parallelize(points)
    val model = LinearRegressionWithSGD.train(pricesRDD,100,0.0000006,1.0,Vectors.zeros(1))
    val prediction = model.predict(Vectors.dense(2500))
    println("Prediction ="+prediction)

  }

}
