#!/usr/bin/python3
# -*- coding: utf-8 -*-


import tensorflow as tf

__author__ = "Iván de Paz Centeno"


class Network(object):

    def __init__(self, session, trainable: bool=True):
        """
        네트워크 초기화
        :param trainable: 네트워크 훈련 가능한지 확인
        """
        self._session = session
        self.__trainable = trainable
        self.__layers = {}
        self.__last_layer_name = None

        with tf.variable_scope(self.__class__.__name__.lower()):
            self._config()

    def _config(self):
        """
        네트워크 계층 구성
        LayerFactory() 클래스 사용해서 수행
        """
        raise NotImplementedError("ERROR")

    def add_layer(self, name: str, layer_output):
        """
        네트워크에 layer 추가
        :param name: 추가할 layer 이름
        :param layer_output: layer output
        """
        self.__layers[name] = layer_output
        self.__last_layer_name = name

    def get_layer(self, name: str=None):
        """
        이름을 기준으로 layer 검색
        :param name: 검색할 layer 이름
        이름이 None이면 네트워크에 마지막으로 추가된 layer 검색
        :return: layer output
        """
        if name is None:
            name = self.__last_layer_name

        return self.__layers[name]

    def is_trainable(self):
        """
        훈련 가능한 flag 가져오기
        """
        return self.__trainable

    def set_weights(self, weights_values: dict, ignore_missing=False):
        """
        네트워크의 가중치 (weight)값 설정
        :param weights_values: 각 layer의 weight 값을 딕셔너리로 저장
        """
        network_name = self.__class__.__name__.lower()

        with tf.variable_scope(network_name):
            for layer_name in weights_values:
                with tf.variable_scope(layer_name, reuse=True):
                    for param_name, data in weights_values[layer_name].items():
                        try:
                            var = tf.get_variable(param_name)
                            self._session.run(var.assign(data))

                        except ValueError:
                            if not ignore_missing:
                                raise

    def feed(self, image):
        """
        이미지를 network에 연결
        :param image: 이미지
        :return: 결과 값 반환
        """
        network_name = self.__class__.__name__.lower()

        with tf.variable_scope(network_name):
            return self._feed(image)

    def _feed(self, image):
        raise NotImplementedError("Method error") #try except