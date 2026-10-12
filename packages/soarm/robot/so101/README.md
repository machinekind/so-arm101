# SO-101 (MJCF)

Model ramienia SO-101 dla MuJoCo, skopiowany bez zmian z
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/ac6b2b09983786f3036cab1000221017fa2193b4/robotstudio_so101)
(commit `ac6b2b0`, licencja Apache-2.0 w pliku [LICENSE](LICENSE)).

Menagerie wywodzi go z `so101_new_calib.xml` z
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101),
czyli z konwencji **nowej kalibracji LeRobota**: zero kazdego stawu w srodku
zakresu. To ta sama konwencja, w ktorej pracuje backend `feetech` i podglad 3D
aplikacji, wiec katy z prawdziwego ramienia ida do modelu bez przesuniec -
tylko stopnie na radiany, a chwytak z 0..100 na kat szczeki. Zgodnosc
sprawdza `tests/test_twin_robots.py` wzgledem niezaleznej kinematyki z Articulusa.

Co menagerie dodalo wzgledem oryginalu: proste ksztalty kolizyjne chwytaka
i ramienia, parametry solvera dobrane pod manipulacje, uchwyt kamery
nadgarstkowej z kamera `wrist_cam`.
