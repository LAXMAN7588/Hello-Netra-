\# Netra



> Assistive Android application for visually impaired users, combining real-time object recognition, OCR, text-to-speech, and emergency assistance.



\## Overview



Netra is an Android-based assistive system designed to provide:



\- Real-time object recognition using the phone camera

\- OCR for reading visible text

\- Text-to-speech feedback

\- Emergency SOS with location sharing

\- Emergency contact management

\- Priority emergency calling

\- Raspberry Pi integration for hardware interaction



The project is being developed as a modular system so that phone-based features can operate independently from Raspberry Pi communication.



\---



\## Current Status



\### Working



\- \[x] Phone camera

\- \[x] Object Recognition

\- \[x] OCR

\- \[x] OCR text-to-speech

\- \[x] SOS

\- \[x] Location acquisition

\- \[x] Emergency contacts

\- \[x] Emergency contact deletion

\- \[x] Priority contact management

\- \[x] Android debug build



\### Integration in progress



\- \[ ] Raspberry Pi signal channel

\- \[ ] Raspberry Pi camera streaming

\- \[ ] Raspberry Pi Object Recognition

\- \[ ] Raspberry Pi OCR

\- \[ ] Physical Raspberry Pi SOS button

\- \[ ] Complete screen-off/background hardware testing



\---



\## Architecture



```text

&#x20;                        NETRA

&#x20;                          |

&#x20;            +-------------+-------------+

&#x20;            |             |             |

&#x20;            v             v             v

&#x20;      Object Detection   OCR           SOS

&#x20;            |             |             |

&#x20;       CameraX        CameraX       Location

&#x20;            |             |             |

&#x20;      ObjectDetector  PaddleOCR       SMS

&#x20;            |             |             |

&#x20;          TTS            TTS       Priority Call





Future Raspberry Pi integration:



&#x20;Raspberry Pi

&#x20;     |

&#x20;     +---- Signal Channel ---------> Android

&#x20;     |          OBJECT

&#x20;     |          OCR

&#x20;     |          SOS

&#x20;     |

&#x20;     +---- Camera UDP :5000 -------> Android

&#x20;                 NETR packets

&#x20;                    |

&#x20;                    v

&#x20;              PiVideoReceiver

&#x20;                    |

&#x20;             +------+------+

&#x20;             |             |

&#x20;         Object          OCR

&#x20;      Recognition

