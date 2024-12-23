## Four Mouse Operations

Mouse operations include: Left Click (`CLICK`), Left Double Click (`DOUBLE_CLICK`), Right Click (`RIGHT_CLICK`), and
Mouse Hover (`HOVER`). For example,
`CLICK(box=[[387,248,727,317]], element_type='Clickable text', element_info='Click to add Title')`. The parameters
supported by these four actions are as follows:

| Parameter Name | Optional | Explanation                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|----------------|----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| box            | Required | Represents a rectangle on the screen in the form `[[a,b,c,d]]`, where a/b/c/d are three-digit numbers ranging from 000 to 999.<br>Assuming the width of the screen is w and the height is h. The top-left corner of the screen is (0, 0). The top-left corner of the rectangle is (a / 1000 * w, b / 1000 * h), and the bottom-right corner of the rectangle is (c / 1000 * w, d / 1000 * h).<br>The actual operation position is the center of the rectangle. |
| element_type   | Optional | A description of the type of the element being operated on, for example, "Clickable text"                                                                                                                                                                                                                                                                                                                                                                      |
| element_info   | Optional | A description of the content of the element being operated on, for example, "Click to add Title"                                                                                                                                                                                                                                                                                                                                                               |

## Text Input (TYPE)

Text input refers to entering text at a given location, for example,
`TYPE(box=[[387,249,727,317]], text='CogAgent', element_type='Text input box', element_info='CogAgent')`. The parameters
it supports are as follows:

| Parameter Name | Optional | Explanation                                                                                                                                                                                                                                                              |
|----------------|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| box            | Required | Please refer to the explanation in "Four Mouse Operations".                                                                                                                                                                                                              |
| element_type   | Optional | Please refer to the explanation in "Four Mouse Operations".                                                                                                                                                                                                              |
| element_info   | Optional | Please refer to the explanation in "Four Mouse Operations".                                                                                                                                                                                                              |
| text           | Required | The text content that needs to be input. This parameter may contain variables in the form `__CogName_xxx__`. During actual execution of the "Text Input" action, these variables should be replaced with actual values. For more details, please refer to [here](#jump). |

## Four Scrolling Operations

Scrolling operations include: Scroll Up (`SCROLL_UP`), Scroll Down (`SCROLL_DOWN`), Scroll Left (`SCROLL_LEFT`), and
Scroll Right (`SCROLL_DOWN`). For example,
`SCROLL_DOWN(box=[[000,086,999,932]], element_type='Scroll', element_info='Scroll', step_count=5)`. The parameters
supported by these four actions are as follows:

| Parameter Name | Optional | Explanation                                                                                                                                                                                                                                                                                       |
|----------------|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| box            | Required | Please refer to the explanation in "Four Mouse Operations".                                                                                                                                                                                                                                       |
| element_type   | Optional | Please refer to the explanation in "Four Mouse Operations".                                                                                                                                                                                                                                       |
| element_info   | Optional | Please refer to the explanation in "Four Mouse Operations".                                                                                                                                                                                                                                       |
| step_count     | Required | The number of steps to scroll, where one step corresponds to one notch of the mouse wheel. Note: Both system settings and application types can affect the actual effect of one step of scrolling, so it is difficult for the model to accurately predict the required number of scrolling steps. |

## Keyboard Press (KEY_PRESS)

Keyboard press refers to pressing and releasing a given button in sequence, for example, `KEY_PRESS(key='F11')`. This
operation type has only one required parameter `key`, which represents the name of the key that needs to be pressed,
such as numeric keys (0–9), letters (A-Z). In addition, `KEY_PRESS` supports the following common keys, as shown in the
table below.

|                              | **Windows**         | **macos**               |
|------------------------------|---------------------|-------------------------|
| **Enter/Return**             | Return              | Return                  |
| **Space**                    | Space               | Space                   |
| **Ctrl key (left/right)**    | Lcontrol / Rcontrol | N/A                     |
| **Alt key (left/right)**     | Lmenu / Rmenu       | N/A                     |
| **Control key (left/right)** | N/A                 | Control / Right Control |
| **Command key (left/right)** | N/A                 | Command / Right Command |
| **Shift key (left/right)**   | Lshift / Rshift     | Shift / Right Shift     |
| **Arrow key - Up**           | Up                  | Up Arrow                |
| **Arrow key - Down**         | Down                | Down Arrow              |
| **Arrow key - Left**         | Left                | Left Arrow              |
| **Arrow key - Right**        | Right               | Right Arrow             |

## Gestures (Combination Keys)

Using combination keys, for example, `ctrl+f` to perform a search. The corresponding structured expression is
`GESTURE(actions=[KEY_DOWN(key='Lcontrol'), KEY_PRESS(key='A'), KEY_UP(key='Lcontrol')])`. `GESTURE` contains only one
parameter `actions`, which takes a list as its value. Each element in the list is one of the following three actions:

1. `KEY_DOWN`: Press a key without releasing it.

2. `KEY_PRESS`: Tap a key, which means to press and release it.

3. `KEY_UP`: Release a key that has been pressed.

## Launching an Application or a URL (LAUNCH)

Directly open an application or a link in a browser. The `LAUNCH` operation accepts two parameters, `app` and `url`,
where `app` represents the name of the application to be opened and `url` represents the link to be opened. If both
parameters are provided, only `url` takes effect. For example:

1. `LAUNCH(app='Settings', url='None')`: Open the system settings.

2. `LAUNCH(app='None', url='baidu.com')`: Open the Baidu homepage.

## Quoting Text Content (QUOTE_TEXT)

Identify and process the text content in a given area, and store the result in a variable for subsequent use. For
example:

1.
`QUOTE_TEXT(box=[[387,249,727,317]], element_type='Text', element_info='Price after coupon: 17.00', output='__CogName_ProductPrice__', result='17.00')`;

2.
`QUOTE_TEXT(box=[[000,086,999,932]], auto_scroll=True, element_type='Window', element_info='CogAgent Technical Report Blog', output='__CogName_TechnicalReport__')`.

The parameters it supports are as follows:

| Parameter Name | Optional | Explanation                                                                                                                                                                                                                                                                                       |
|----------------|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| box            | Required | Please refer to the explanation in "Four Mouse Operations".                                                                                                                                                                                                                                       |
| element_type   | Optional | Please refer to the explanation in "Four Mouse Operations".                                                                                                                                                                                                                                       |
| element_info   | Optional | Please refer to the explanation in "Four Mouse Operations".                                                                                                                                                                                                                                       |
| output         | Required | The variable name indicating where the quoted result is stored. The format is `__CogName_xxx__`.                                                                                                                                                                                                  |
| result         | Optional | Represents the result of the text quote. If the text quote result is too long, the value of `result` will contain ellipses, or this parameter may not be present at all. In such cases, the CogAgent client-side application needs to call an OCR service to obtain the quoted result.            |
| auto_scroll    | Optional | Defaults to `False`. If `auto_scroll` is true, the CogAgent client-side application needs to scroll down the list until the bottom of the list, while obtaining the list content as the result of the medical text. When the text to be quoted is very long, `auto_scroll` should be set to true. |

## Calling a Large Language Model (LLM) and Using Variables

Organize prompts and call a large language model to compute results. The parameters received by this action are as
follows:

| Parameter Name | Optional | Explanation                                                                                                                                                                                                                                                                                           |
|----------------|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| prompt         | Required | The prompt used to call the large language model, where the variable names used will be replaced with actual values.                                                                                                                                                                                  |
| output         | Required | The variable name indicating where the result of the large language model call is stored. The format is `__CogName_xxx__`.                                                                                                                                                                            |
| result         | Optional | Represents the result of the large language model call. If the result is too long, the value of `result` will contain ellipses, or this parameter may not be present at all. In such cases, the CogAgent client-side application needs to call the large language model service to obtain the result. |

For example, the following two operations can be used to summarize the entire content of this page:

1. Quote the entire content of this page. Since there is a lot of content on the page, `auto_scroll=True` needs to be
   set when quoting.

    1. `QUOTE_TEXT(box=[[000,086,999,932]], auto_scroll=True, element_type='Window', element_info='CogAgent Technical Report Blog', output='__CogName_TechnicalReport__')`

2. <span id='jump'>Call</span> the large language model to summarize the content of the technical report. The content of
   the technical report has been stored in the variable `__CogName_TechnicalReport__`, so this variable should be used
   directly in the `prompt` parameter; when calling the large language model to generate the summary content,
   `__CogName_TechnicalReport__` needs to be replaced with the actual value.

    1. `LLM(prompt='Summarize the following content: __CogName_TechnicalReport__', output='__CogName_TechnicalReportSummary__')`

## Quoting Clipboard Content (QUOTE_CLIPBOARD)

Store the content of the clipboard in a variable for use in subsequent steps. Many web pages and applications provide
a "click to copy to clipboard" feature. `QUOTE_CLIPBOARD` allows the model to quickly obtain and use the content of the
clipboard.
Here is an example of a structured expression:
`QUOTE_CLIPBOARD(output='__CogName_QuickSortCode__', result='def quick_sort(arr):\n\tif len(arr) <= 1:\n\t\treturn arr\n\t...')`

## End (END)

A special operation indicating that the task has been completed.

