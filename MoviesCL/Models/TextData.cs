using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace MoviesCL.Models
{
    public class TextData
    {
        [LoadColumn(0)]
        public string Id { get; set; }
        [LoadColumn(1)]
        public string Title { get; set; }
        [LoadColumn(2)]
        public string Plot { get; set; }
    }
}
