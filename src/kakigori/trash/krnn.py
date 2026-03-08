                    tk.getHumdrumFile(str(out_krn))
                    # Get the Humdrum data as a string instead of saving directly to a file
                    with open(out_krn, "r", encoding="utf-8") as f:
                        humdrum_data = f.read()

                    # 1. Fix the negative tuplet bug (e.g., changes "32%-3" to "32%3")
                    #humdrum_data = re.sub(r'%-(\d+)', r'%\1', humdrum_data)

                    # 2. Strip the .ZZZ artifacts
                    humdrum_data = humdrum_data.replace('.ZZZ', '')

                    # 3. Fix the parser-breaking negative tuplet (changes "32%-3" to "32%3")
                    humdrum_data = humdrum_data.replace('%-', '%')
                    # Save the cleaned data to your output path
                    with open(out_krn, "w", encoding="utf-8") as f:
                        f.write(humdrum_data)