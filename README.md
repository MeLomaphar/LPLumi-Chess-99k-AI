 # LPLumi Chess AI (99k params)
 
I trained a recurrent AI model to play chess.
LPLumi Chess 99k (Lomaphar-Lumi_Chess)

Model: 99,400 parameters (about 400 KB). A very small model relative to its capabilities.

Under the hood:
Recursive depth, a 4-loop attention block, allows the network to think beyond its weights (verified on loss and Stockfish as benchmarks).
Trained on: Stockfish dataset + distillation on Magnus Carlsen games.
Alpha-Beta search + position evaluation.

Testing against Stockfish showed 1350 Elo.
To run: pip install torch chess numpy

(I’m a bad chess player; it beats me well, especially toward the end. At the beginning it blunders pieces, so you can play aggressively.)

How to run
Installing dependencies
bash
pip install torch chess numpy
Top-1 Accuracy (vs Stockfish): ~46%

Estimated Elo: 1350 (based on match results against Stockfish with a UCI_Elo limit).
 
 
 
 # LPLumi Chess AI (99k params)

Обучил рекуррентную ИИ модель играть в шахматы.
LPLumi Chess 99k (Lomaphar-Lumi_Chess)

Модель: 99400 параметров (около 400кб). Очень маленькая модель по относительно способностей.

Под капотом:
Рекурсивная глубина, блок внимания 4 loops, позволяет сети думать не на свои веса (Проверено на loss и stockfish в качестве бенчмарков).
Обучена: Датасет Stockfish + дистиляция на партиях Магнуса Карлсена.
Alpha-Beta поиска + оценка позиции.

Тест показал на StockFish показал 1350 elo
Для запуска: pip install torch chess numpy

(Я плохой игрок в шахматы, меня уделывает хорошо, особенно ближе к концу, в начале зевает фигуры можно играть агрессивно)

## Как запустить

### Установка зависимостей
bash
pip install torch chess numpy

Top-1 Accuracy (vs Stockfish): ~46%

Оценка Elo: 1350 (по результатам матча против Stockfish с ограничением UCI_Elo).
