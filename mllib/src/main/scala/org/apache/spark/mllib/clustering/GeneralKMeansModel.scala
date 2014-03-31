/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.clustering

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._

import org.apache.spark.mllib.linalg.Vector

import breeze.linalg.{DenseVector => BDV, Vector => BV, norm => breezeNorm}

/**
 * A clustering model for K-means. Each point belongs to the cluster with the closest center.
 */
class GeneralKMeansModel(val measure: BregmanDivergenceMeasure.DivergenceFunc, val clusterCenters: Array[BV[Double]]) extends Serializable {
  /** Total number of clusters. */
  def k: Int = clusterCenters.length

  /** Return the cluster index that a given point belongs to. */
  def predict(point: Vector): Int = {
    GeneralKMeans.findClosest(measure, clusterCenters, point.toBreeze)._1
  }

  /**
   * Return the K-means cost (sum of squared distances of points to their nearest center) for this
   * model on the given data.
   */
  def computeCost(data: RDD[Vector]): Double = {
    data.map(p => GeneralKMeans.pointCost(measure, clusterCenters, p.toBreeze)).sum()
  }
}
