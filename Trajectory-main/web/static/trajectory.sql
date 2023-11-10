/*
 Navicat Premium Data Transfer

 Source Server         : ma
 Source Server Type    : MySQL
 Source Server Version : 80017
 Source Host           : localhost:3306
 Source Schema         : trajectory

 Target Server Type    : MySQL
 Target Server Version : 80017
 File Encoding         : 65001

 Date: 13/10/2022 22:47:52
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for trajectory
-- ----------------------------
DROP TABLE IF EXISTS `trajectory`;
CREATE TABLE `trajectory`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `length` int(11) NOT NULL,
  `start_time` int(11) NOT NULL,
  `end_time` int(11) NOT NULL,
  `points` text CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `embedding` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 5508 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Compact;

-- ----------------------------
-- Records of trajectory
-- ----------------------------
INSERT INTO `trajectory` VALUES (1, 2, 1478063520, 1478063530, '[(124.2541,92.12354),(125.1254,84.25613)]', '[-0.15446665111254786,0.12365478952136547]');
INSERT INTO `trajectory` VALUES (2, 2, 1478063600, 1478063750, '[(124.2881,92.128654),(125.1864,84.87613)]', '[-0.1544665781254786,0.1236547895255547]');

SET FOREIGN_KEY_CHECKS = 1;
