package org.apache.spark.mllib.clustering

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.mllib.linalg.{Vector, Vectors}

import org.apache.spark.rdd.RDD

trait KMeansModel {
  def predict(point: Vector): Int
  
  /** Maps given points to their cluster indices. */
  def predict(points: RDD[Vector]): RDD[Int]

  /**
   * Return the K-means cost (sum of squared distances of points to their nearest center) for this
   * model on the given data.
   */
  def computeCost(data: RDD[Vector]): Double 
}