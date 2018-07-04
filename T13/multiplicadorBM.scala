import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD


object MultiplicadorBM {

  // Args = path/to/text0.txt path/to/text1.txt
  def main(args: Array[String]) {

    // create Spark context with Spark configuration
    val sc = new SparkContext(new SparkConf().setAppName("Multiplicacao de Matrizes"))

    // Carrega o arquivo contendo o formato i j valor e converte pra tupla de entrada
    val matrix_input1 = sc.textFile(args(0)).map(line => line.split(' ')).map{e => (e(0).toLong, e(1).toLong, e(2).toDouble)}
    
    // Gera os entries e Constroi a CoordinateMatrix 
    val matrix_matrixEntry1 = matrix_input1.map(e => MatrixEntry(e._1, e._2, e._3))
    val data1 = new CoordinateMatrix(matrix_matrixEntry1)
    val local_data1 = data1.toBlockMatrix(data1.numRows.toInt/2, data1.numCols.toInt/2)
    
    // println("Matriz 1")
    // val local_data1 = data1.toBlockMatrix.toLocalMatrix
    // val m1 = local_data1.numRows
    // val n1 = local_data1.numCols
    // for (i <- 0 to m1-1){
    //     for (j <- 0 to n1-1){
    //         print(" "+local_data1.apply(i,j))
    //     }
    //     println()
    // }

    val matrix_input2 = sc.textFile(args(1)).map(line => line.split(' ')).map{e => (e(0).toLong, e(1).toLong, e(2).toDouble)}
    
    val matrix_matrixEntry2 = matrix_input2.map(e => MatrixEntry(e._1, e._2, e._3))
    val data2 = new CoordinateMatrix(matrix_matrixEntry2)
    val local_data2 = data2.toBlockMatrix(data2.numRows.toInt/2, data2.numCols.toInt/2)
    
    // println("Matriz 2")
    // val local_data2 = data2.toBlockMatrix.toLocalMatrix
    // val m2 = local_data2.numRows
    // val n2 = local_data2.numCols
    // for (i <- 0 to m2-1){
    //     for (j <- 0 to n2-1){
    //         print(" "+local_data2.apply(i,j))
    //     }
    //     println()
    // }

    //Multiplica as matrizes 
    // val M_ = data1.entries.map({ case MatrixEntry(i, j, v) => (j, (i, v)) })
    // val N_ = data2.entries.map({ case MatrixEntry(j, k, w) => (j, (k, w)) })
    // val productEntries = M_.join(N_).map({ case (_, ((i, v), (k, w))) => ((i, k), (v * w)) }).reduceByKey(_ + _).map({ case ((i, k), sum) => MatrixEntry(i, k, sum) })
    
    // val saida = new CoordinateMatrix(productEntries)

    val saida = local_data1.multiply(local_data2)

    //Imprime a matriz resultante
    val local_mat = saida.toLocalMatrix
    val m = local_mat.numRows
    val n = local_mat.numCols
    for (i <- 0 to m-1){
        for (j <- 0 to n-1){
            print(" "+local_mat.apply(i,j))
        }
        println()
    }
    System.exit(1)
  }

}