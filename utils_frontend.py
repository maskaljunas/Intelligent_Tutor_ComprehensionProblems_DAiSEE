"""
Simple Diaglog interface for presenting and requesting information.
"""

import tkinter as tk
from tkinter import simpledialog


def get_user_info():
    root = tk.Tk()
    root.withdraw()

    # subject
    subject = simpledialog.askstring(title="Studie zur Untersuchung von Verständnisproblemen beim Lernen",
                                      prompt="Hallo, vielen Dank, dass du an meiner Studie teilnimmst! :-)\n"
                                             "Zuerst werden ein paar Fragen zur Person gestellt. \n"
                                             "Bitte trage diese unten ein und bestätige mit der Eingabe-Taste oder "
                                             "drücke auf OK.\n\n\nVP-Name: \nz.B Tom Müller --> tm")

    # gender
    gender = simpledialog.askstring(title="Studie zur Untersuchung von Verständnisproblemen beim Lernen",
                                      prompt="Geschlecht?  \nw oder m")

    # age
    age = simpledialog.askstring(title="Studie zur Untersuchung von Verständnisproblemen beim Lernen",
                                      prompt="Alter in Jahren")


    # education
    education = simpledialog.askstring(title="Studie zur Untersuchung von Verständnisproblemen beim Lernen",
                                      prompt="Höchster Bildungsabschluss: \n\n"
                                             "1: Hauptschulabschluss\n2: Realschulabschluss\n"
                                             "3: Abitur\n4: Abgeschlossene Ausbildung\n"
                                             "5: Fachhochschulabschluss\n6: Bachelor \n"
                                             "7: Master oder Diplom \n8: Promotion")

    ready = simpledialog.askstring(title="Studie zur Untersuchung von Verständnisproblemen beim Lernen",
                                   prompt="Dir wird gleich in einem kurzen Video erklärt wie der einfache Plural der "
                                          "Finnischen Sprache gebildet wird.\nPass genau auf und versuche das Gezeigte"
                                          " zu verstehen und einzuprägen, denn am Ende wird ein kleiner Test"
                                          " durchgeführt.\n\nAm Ende solltest du den Plural von 8 Wörtern bilden "
                                          "können.\n\n\nBist du bereit? j/n")
    user_data = [subject, gender, age, education]

    return user_data, ready


def help_request():
    root = tk.Tk()
    root.withdraw()

    answer = simpledialog.askstring(title="Studie zur Untersuchung von Verständnisproblemen beim Lernen",
                                      prompt="Der Tutor hat anhand deiner Gesichtsreaktion gemerkt, \n"
                                             "dass du ein wenig Verständnisprobleme hast. \n"
                                             "Willst du eine Zusammenfassung des Erklärten noch einmal sehen?\n\n"
                                             "Ja (j) / Nein (n)")
    return answer


