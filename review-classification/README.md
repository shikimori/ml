# [Shikimori] Review Classificator

_Подробная документация в процессе_

## 1, 2, СТАРТ

Необходимы:

- `Python >= 3.8`
- `tensorflow >= 2.0`

Для обладателей макбуков на М1/М2 процессорах: `tensorflow-macos >= 2.0`

### Устанавливаем модули

```bash
$> pip install -r requirements.txt 
```

### Подготавливаем данные для обучения

1. Если выражать подготовку SQL-ем, то запрос примерно такой:

```sql
    select id,
           body,
           opinion
    from reviews
    where (opinion = 'neutral' and actually_neutral = True)
       or (opinion != 'neutral')
```

где:

- `id` - уникальный индетификатор, `integer`
- `body` - текст отзыва, `string`
- `opinion` - окрас отзыва (`positive`, `negative`, `neutral`)
- `actually_neutral` - в данном случае - условный показатель того, что отзыв **действительно** нейтральный. Ведь "вроде
  нейтральные" мы и будем предсказывать

2. Сохраняем в `<repository>/demo/input.json`. JSON-схема выглядит примерно так:

```json
{
  "$schema": "http://json-schema.org/draft-04/schema#",
  "type": "array",
  "items": [
    {
      "type": "integer"
    },
    {
      "type": "string"
    },
    {
      "type": "string"
    }
  ]
}
```

т.е. данные должны быть в ввиде массива массивов, пример:

```json
[
  [
    14142, // 'id'
    "<text>", // 'body'
    "negative" // 'opinion'
  ],
  [
    14145,
    "<text>",
    "negative"
  ]
]
```

### Подготавливаем данные для предсказаний (которые мы хотим разметить классификатором)

1. SQL-подобный запрос:

```sql
    select id,
           body
    from reviews
    where opinion = 'neutral'
      and actually_neutral = False
```

2. Сохраняем в похожей схеме, только без `opinion` в `<repository>/demo/prod.json`

```json
[
  [
    14143,
    "<text>"
  ],
  [
    14146,
    "<text>"
  ]
]
```

### Запускаем обучение и предсказание

```bash
$> make pipeline
```

### Получаем результаты

В формате:

```json
{
  "global_id": 19,
  "p_target": "positive"
}
```

В файле `<repository>/demo/result.json`