/**
 * @author Derrick Burns <derrick.burns@rincaro.com>
 *
 */
package org.apache.spark.mllib.clustering

import breeze.linalg.{DenseVector => BDV, Vector => BV, norm => breezeNorm}


/**
 * See  http://en.wikipedia.org/wiki/Bregman_divergence
 */
object BregmanDivergenceMeasure {
  
  type DivergenceFunc = (BV[Double],BV[Double]) => Double
  
  /**
   * http://en.wikipedia.org/wiki/Euclidean_distance
   */
  def euclideanDistanceSquared(p: BV[Double], c: BV[Double]): Double = {
    val point = p.toArray
    val center = c.toArray
    (point, center).zipped.foldLeft(0.0) {
      case (d, (p, c)) => d + (p - c) * (p - c)
    }
  }
  
  /**
   * http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
   */
  def kullbackLeibler(p: BV[Double], c: BV[Double]): Double = {
    val point = p.toArray
    val center = c.toArray
    (point, center).zipped.foldLeft(0.0) {
      case (d, (p, c)) if p != 0.0 && c != 0.0 => d + p * Math.log(p / c) 
      case (d, _) => d
    }
  }

  def generalizedKullbackLeibler(p: BV[Double], c: BV[Double]): Double = {
    val point = p.toArray
    val center = c.toArray
    (point, center).zipped.foldLeft(0.0) {
      case (d, (p, c)) if p != 0.0 && c != 0.0 => d + p * Math.log(p / c) - p + c 
      case (d, _) => d
    }
  }


/**
 * projectiveKullbackLeibler is a generalization of Kullback-Leibler Divergence obtained by performing L1Normalization
 * of the vectors. 
 * 
 *   projectiveKullbackLeibler( v,w ) = ||v1|| * kullbackLeibler( v / ||v||, w / ||w|| ), where ||v|| = L1 norm of v = sum(i) v_i
 * 
 * This measure is appropriate when the values input values are raw frequencies.  L1Normalization 
 * transforms them into points on the unit simplex.  
 * 
 * In this measure, the distance between (x, y, z) and c * (x,y,z) is zero since: 
 *     (x,y,z) / ||(x,y,z)||  equals c(x,y,z) / ||c(x,y,z)|| under the L1 norm
 * 
 */
 
  def projectiveKullbackLeibler(p: BV[Double], c: BV[Double]): Double = {
    val point = p.toArray
    val center = c.toArray
    val parts = (point, center).zipped.foldLeft((0.0,0.0,0.0)) {
      case ((d,psum,csum), (p, c)) if p != 0.0 && c != 0.0 => (d + p * Math.log(p / c), psum + p, csum + c)
      case ((d,psum,csum), (p, c))  => (d, psum + p, csum + c)
    }
    parts match {
      case (d,psum,csum) if psum != 0.0 && csum != 0.0 => d - psum * Math.log(psum / csum)
      case (d, _, _) => d
    }
  }


  /**
   * http://en.wikipedia.org/wiki/Itakura%E2%80%93Saito_distance
   */
  def itakuraSaito(p: BV[Double], c: BV[Double]): Double = {
    val point = p.toArray
    val center = c.toArray
    (point, center).zipped.foldLeft(0.0) {
      case (d, (p, c)) if p !=0 && c != 0 => d + (p / c) - Math.log( p / c) - 1
      case (d, _)  => d
    }
  }
}