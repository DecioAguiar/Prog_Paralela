import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf


object Analisador {

  // Args = path/to/text0.txt path/to/text1.txt
  def main(args: Array[String]) {

    // create Spark context with Spark configuration
    val sc = new SparkContext(new SparkConf().setAppName("Contagem de Palavra"))

    println("TEXT1")

    // read first text file and split into lines
    val lines1 = sc.textFile(args(0))
    val words1 = lines1.flatMap(line => line.split (" ")).map(word => word.replaceAll("[,.:;!?]", "").toLowerCase)
    // cada item do RDD é uma palavra do arquivo
    val intermData1 = words1.filter(word => word.length() > 3)
    // cada item do arquivo é um par (palavra,1)
    val wordCount1 = intermData1.map(word => (word,1)).reduceByKey{case(x,y) => x+y}
    // cada item do RDD contém ocorrência final de cada palavra
    val contagens1 = wordCount1.takeOrdered(5)(Ordering[Int].reverse.on(x => x._2))
    // 5 resultados no programa driver
    contagens1.foreach(x => println(x._1+"="+x._2))
    // TODO: contar palavras do texto 1 e imprimir as 5 palavras com as maiores ocorrencias (ordem DECRESCENTE)
    // imprimir na cada linha: "palavra=numero"

    println("TEXT2")

    // read second text file and split each document into words
    val lines2 = sc.textFile(args(1))
    val words2 = lines2.flatMap(line => line.split (" ")).map(word => word.replaceAll("[,.:;!?]", "").toLowerCase)
    // cada item do RDD é uma palavra do arquivo
    val intermData2 = words2.filter(word => word.length() > 3)
    // cada item do arquivo é um par (palavra,1)
    val wordCount2 = intermData2.map(word => (word,1)).reduceByKey{case(x,y) => x+y}
    // cada item do RDD contém ocorrência final de cada palavra
    val contagens2 = wordCount2.takeOrdered(5)(Ordering[Int].reverse.on(x => x._2))
    // 5 resultados no programa driver
    contagens2.foreach(x => println(x._1+"="+x._2))
    // TODO: contar palavras do texto 2 e imprimir as 5 palavras com as maiores ocorrencias (ordem DECRESCENTE)
    // imprimir na cada linha: "palavra=numero"

    println("COMMON")
    val max1 = wordCount1.filter(x => x._2 > 100)
    val max2 = wordCount2.filter(x => x._2 > 100)

    val resultRdd = max1.join(max2).map(r => (r._1, r._2._1)).distinct()
    val resultado = resultRdd.sortByKey().collect()
    resultado.foreach(x => println(x._1))

    // TODO: comparar resultado e imprimir na ordem ALFABETICA todas as palavras que aparecem MAIS que 100 vezes nos 2 textos
    // imprimir na cada linha: "palavra"

  }

}