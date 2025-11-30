# Accelerating Median Filtering For Image Noise Reduction

Final Project  
CpE 411 — Data Structures and Algorithms  
BS Computer Engineering  
Batangas State University–TNEU

## Project Title
Accelerating Median Filtering For Image Noise Reduction: An Integration Of Tree-Based Arrays And Priority Queue Pairs

## Introduction / Background
The proposed system is a high-performance algorithmic tool designed to rapidly eliminate noise from digital images by using a specialized "tri-core" data structure architecture. Unlike traditional denoising methods that suffer from latency due to repetitive sorting, this system integrates 2D Arrays for direct storage, Priority Queue Pairs (Min-Max Heaps) for dynamic median calculation, and Range Trees for efficient color logic, further optimized by a Dictionary cache. This integration effectively bypasses the processing bottlenecks of standard filters, allowing for cleaner images without the usual computational lag.

This system is needed to address the efficiency gap in modern image processing, where high-resolution data often cripples standard sorting algorithms. It primarily benefits computer vision engineers, medical imaging professionals, and developers working with images or real-time video feeds, all of whom require restoration tools that are both mathematically accurate and computationally inexpensive. By drastically reducing the time complexity of the filtering process, this project provides a scalable solution for users who need high-fidelity results on hardware with limited processing power.

## Problem Statement
The fundamental challenge in the field of digital image processing is the effective removal of impulse noise without compromising the structural integrity of the image. While the standard median filter is widely accepted as the optimal solution for preserving edges during restoration, its application is severely hindered by computational inefficiency. The traditional implementation of this filter relies on a brute-force approach that necessitates sorting pixel values within a sliding kernel window for every coordinate in the image grid. As the resolution of the image or the size of the kernel increases, this repetitive sorting operation creates a substantial processing bottleneck, commonly referred to as the sorting overhead.

A critical technical limitation lies in the data management strategies of existing denoising systems, which often utilize generic linear data structures to handle distinct computational tasks. By treating storage, median calculation, and range querying as uniform operations, these systems suffer from structural redundancy. Specifically, the inability to dynamically track median values as the window slides results in unnecessary recalculations of overlapping data, while the lack of hierarchical spatial partitioning leads to inefficient memory access patterns during color quantization.

Consequently, these architectural inefficiencies render standard median filtering methods unsuitable for high-performance environments. The excessive time complexity required to process high-resolution data creates unacceptable latency for real-time applications, such as live video feeds or rapid medical diagnostics. This performance gap necessitates a system that moves beyond simple algorithmic implementation and focuses on optimizing the underlying data structures to handle the specific demands of image restoration.

## Objectives
To design, implement, and rigorously test an Optimized Image Denoiser System that significantly reduces the time complexity of the Median Filtering algorithm by leveraging specialized data structures for high-performance noise reduction in digital images. Specifically, it aims to:

1. Implement the three core complex data structures, the Priority Queue Pair (Min/Max Heaps), the Segment Tree (Tree-Based Array), and the 2D Array, to correctly manage and process image pixel data.
2. Develop and integrate the Optimized Sliding Window Algorithm that utilizes the Priority Queue Pair to calculate the median pixel value and update the window in O(log N) time complexity.
3. Implement the Segment Tree to efficiently track and query the distribution of pixel values within the sliding window, ensuring range-based information needed for advanced filtering or optimization can be retrieved in O(log N) time.
4. Quantify the acceleration achieved and verify the O(log N) time complexity by testing the optimized system against a standard (brute-force) Median Filter across various image sizes.
5. Process and denoise sample images corrupted by "salt-and-pepper" noise, qualitatively demonstrating the effectiveness of the optimized system in preserving image features while achieving superior processing speed.

## Course Requirements Alignment
- Applies at least **three data structures**: 2D arrays, dual heaps, segment tree, dictionary cache.
- Implements at least **two algorithms**: optimized sliding-window median maintenance and Fenwick-tree range queries (plus brute-force baseline for benchmarking).
- Demonstrates both batch and pseudo real-time processing via CLI + notebook workflows.
- Targets measurable performance optimization compared to naive approaches.
